from fastapi import FastAPI, UploadFile, File
from typing import List, Dict, Any, Tuple
import os
from PIL import Image, ImageChops, ImageEnhance, ExifTags
import exifread
import numpy as np
import cv2
import tempfile
import shutil
import hashlib
import imagehash
import matplotlib.pyplot as plt
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import piexif
from hachoir.metadata import extractMetadata
from hachoir.parser import createParser
from datetime import datetime
import io
from skimage import restoration, img_as_float
from scipy import signal
import warnings
import logging
import subprocess
import pywt
import base64
import mediapipe as mp
from ultralytics import YOLO

# --- PRNU: imports ---
try:
    import pywt
except ImportError:
    raise ImportError("PyWavelets not found. Install with: pip install PyWavelets")

import base64
yolo_model = YOLO("yolov8n.pt")

app = FastAPI()

app.mount("/prnu_maps", StaticFiles(directory="prnu_maps"), name="prnu")

import os

media_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "media"))
app.mount("/media", StaticFiles(directory=media_dir), name="media")

# Mount folders for static file access
app.mount("/ela_results", StaticFiles(directory="ela_results"), name="ela")
app.mount("/lighting_maps", StaticFiles(directory="lighting_maps"), name="heatmap")
temp_dir = tempfile.gettempdir()
app.mount("/tmp", StaticFiles(directory=temp_dir), name="tmp")
app.mount("/copy_move_maps", StaticFiles(directory="copy_move_maps"), name="copymove")
app.mount("/images", StaticFiles(directory="images"), name="images")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def root():
    return {"message": "Image Sentiment API is up and running"}

# ===== Metadata Extraction =====
def extract_pil_metadata(image_path: str) -> Dict[str, Any]:
    metadata = {}
    try:
        image = Image.open(image_path)
        info = image._getexif()
        if info:
            for tag, value in info.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                metadata[str(tag_name)] = str(value)
        else:
            metadata['Info'] = "No EXIF metadata found using PIL."
    except Exception as e:
        metadata['Error'] = f"Error reading image with PIL: {e}"
    return metadata

def extract_exifread_metadata(image_path: str) -> Dict[str, Any]:
    metadata = {}
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)
            if tags:
                for tag in tags.keys():
                    metadata[str(tag)] = str(tags[tag])
            else:
                metadata['Info'] = "No EXIF metadata found using exifread."
    except Exception as e:
        metadata['Error'] = f"Error reading image with exifread: {e}"
    return metadata

def extract_piexif_metadata(image_path: str) -> Dict[str, Any]:
    metadata = {}
    try:
        exif_dict = piexif.load(image_path)
        for ifd in exif_dict:
            for tag in exif_dict[ifd]:
                tag_name = piexif.TAGS[ifd][tag]["name"] if tag in piexif.TAGS[ifd] else tag

                value = exif_dict[ifd][tag]
                try:
                    if isinstance(value, bytes):
                        value = value.decode(errors='ignore')
                except Exception:
                    pass
                metadata[f"{ifd}:{tag_name}"] = str(value)
        if not metadata:
            metadata['Info'] = "No EXIF metadata found using piexif."
    except Exception as e:
        metadata['Error'] = f"Error reading image with piexif: {e}"
    return metadata

def extract_all_metadata(image_path: str) -> Dict[str, Any]:
    return {
        "PIL Metadata": extract_pil_metadata(image_path),
        "ExifRead Metadata": extract_exifread_metadata(image_path),
        "Piexif Metadata": extract_piexif_metadata(image_path)
    }
def extract_jpeg_structure_metadata(image_path: str) -> dict:
    try:
        print(f"[DEBUG] Trying to parse: {image_path}")
        parser = createParser(image_path)
        if not parser:
            print("[DEBUG] Parser could not be created")
            return {"Error": f"Could not parse image: {image_path}"}

        with parser:
            metadata = extractMetadata(parser)
        
        if not metadata:
            print("[DEBUG] Metadata extraction returned None")
            return {"Error": "No metadata extracted"}
        
        result = {}
        for item in metadata.exportPlaintext():
            # item is like "- Duration: 2s"
            key_value = item.strip("- ").split(": ", 1)
            if len(key_value) == 2:
                key, value = key_value
                result[key] = value
        
        print(f"[DEBUG] Extracted metadata: {result}")
        return result

    except Exception as e:
        print(f"[ERROR] Failed to extract metadata: {e}")
        return {"Error": str(e)}
def extract_digest_info(image_path: str) -> dict:
    try:
        digest_info = {}

        # Filename
        digest_info["Filename"] = os.path.basename(image_path)

        # Filetime (last modified)
        mtime = os.path.getmtime(image_path)
        digest_info["Filetime"] = datetime.utcfromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S GMT')

        # File size
        digest_info["File Size"] = f"{os.path.getsize(image_path):,} bytes"

        # File type
        file_ext = os.path.splitext(image_path)[-1].lower()
        if file_ext == '.jpg' or file_ext == '.jpeg':
            file_type = 'image/jpeg'
        else:
            file_type = f"unknown ({file_ext})"
        digest_info["File Type"] = file_type

        # Image info
        with Image.open(image_path) as img:
            digest_info["Dimensions"] = f"{img.width}x{img.height}"
            digest_info["Color Channels"] = len(img.getbands())
            colors = img.getcolors(maxcolors=2**24)
            if colors:
                digest_info["Unique Colors"] = len(colors)
            else:
                digest_info["Unique Colors"] = "Too many to count"

        # Hashes
        with open(image_path, "rb") as f:
            data = f.read()
            digest_info["MD5"] = hashlib.md5(data).hexdigest()
            digest_info["SHA1"] = hashlib.sha1(data).hexdigest()
            digest_info["SHA256"] = hashlib.sha256(data).hexdigest()

        # First analyzed (current time)
        digest_info["First Analyzed"] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S GMT')

        return digest_info

    except Exception as e:
        return {"Error": str(e)}
def get_jpeg_quality_details(image_path):
    try:
        with Image.open(image_path) as img:
            if img.format != "JPEG":
                return {"quality_info": "Not a JPEG image"}
            
            quant_tables = img.quantization  # Dictionary of tables
            q_tables_info = {}

            for table_id, table in quant_tables.items():
                matrix = [table[i:i + 8] for i in range(0, len(table), 8)]
                label = "Luminance" if table_id == 0 else f"Chrominance {table_id}"
                q_tables_info[label] = matrix

            # You can estimate quality based on the luminance table (Q0)
            # This is a rough estimate and not standardized
            luminance = quant_tables.get(0, [])
            estimated_quality = "Unknown"
            if luminance:
                q_sum = sum(luminance)
                if q_sum < 300:
                    estimated_quality = "High (~95-100%)"
                elif q_sum < 500:
                    estimated_quality = "Medium (~75-95%)"
                else:
                    estimated_quality = "Low (<75%)"

            return {
                "quality_estimate": estimated_quality,
                "quantization_tables": q_tables_info
            }

    except Exception as e:
        return {"error": str(e)}

# ===== ELA Analysis =====
def perform_ela(image_path: str, ela_output_folder="ela_results", quality=90) -> str:
    try:
        os.makedirs(ela_output_folder, exist_ok=True)
        original = Image.open(image_path).convert('RGB')
        temp_path = os.path.join(ela_output_folder, "temp.jpg")
        original.save(temp_path, 'JPEG', quality=quality)
        resaved = Image.open(temp_path)
        ela_image = ImageChops.difference(original, resaved)
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max_diff if max_diff != 0 else 1
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        ela_filename = os.path.splitext(os.path.basename(image_path))[0] + "_ela.png"
        ela_output_path = os.path.join(ela_output_folder, ela_filename)
        ela_image.save(ela_output_path)
        return ela_output_path
    except Exception as e:
        return f"Error during ELA for {image_path}: {e}"

# ===== Splicing ELA =====
def generate_splicing_ela(image_path: str, output_folder="ela_results", quality=90) -> str:
    try:
        os.makedirs(output_folder, exist_ok=True)
        original = Image.open(image_path).convert('RGB')
        temp_path = os.path.join(output_folder, "splicing_temp.jpg")
        original.save(temp_path, 'JPEG', quality=quality)
        compressed = Image.open(temp_path)
        ela_image = ImageChops.difference(original, compressed)
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max_diff if max_diff != 0 else 1
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        filename = os.path.splitext(os.path.basename(image_path))[0] + "_splicing_ela.png"
        ela_path = os.path.join(output_folder, filename)
        ela_image.save(ela_path)
        return f"ela_results/{filename}"
    except Exception as e:
        return f"Error generating splicing ELA: {e}"

# ===== Lighting Analysis =====
def analyze_lighting_inconsistencies(image_path: str, output_folder="lighting_maps") -> Dict[str, Any]:
    result = {}
    try:
        os.makedirs(output_folder, exist_ok=True)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            return {"error": "Could not read image with OpenCV."}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel_size = 15
        mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        mean_sq = cv2.blur((gray.astype(np.float32) ** 2), (kernel_size, kernel_size))
        variance = mean_sq - (mean ** 2)
        result["mean_local_variance"] = float(np.mean(variance))
        result["std_local_variance"] = float(np.std(variance))
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        result["brightness_histogram"] = hist.flatten().tolist()
        norm_variance = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm_variance, cv2.COLORMAP_JET)
        heatmap_filename = os.path.splitext(os.path.basename(image_path))[0] + "_heatmap.png"
        heatmap_path = os.path.join(output_folder, heatmap_filename)
        cv2.imwrite(heatmap_path, heatmap)
        result["heatmap_path"] = heatmap_path
    except Exception as e:
        result["error"] = f"Lighting analysis failed: {e}"
    return result

# ===== Hashing =====
def calculate_hash(file_path, algo='md5'):
    with open(file_path, "rb") as f:
        data = f.read()
    if algo == 'md5':
        return hashlib.md5(data).hexdigest()
    return hashlib.sha256(data).hexdigest()

def perceptual_hash(file_path):
    img = Image.open(file_path)
    return str(imagehash.phash(img))

# ===== Noise Analysis =====
def visualize_noise(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    noise_map = cv2.absdiff(img, blurred)
    noise_image_path = os.path.splitext(image_path)[0] + "_noise.png"
    plt.imsave(noise_image_path, noise_map, cmap='gray')
    return f"tmp/{os.path.relpath(noise_image_path, start=tempfile.gettempdir())}"

def region_noise_variation(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h,w = img.shape [:2] 
    results = []
    for y in range(2):
        for x in range(2):
            region = img[y*h//2:(y+1)*h//2, x*w//2:(x+1)*w//2]
            results.append({f"region_{y}_{x}": round(float(np.std(region)), 2)})
    return results

def lighting_histogram(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist, _ = np.histogram(hsv[:, :, 2].ravel(), bins=256, range=[0, 256])
    return hist.tolist()

# ===== Copy-Move Forgery =====
def detect_copy_move(image_path, output_folder="copy_move_maps", block_size=32, stride=16, threshold=0.9):
    result = {}
    try:
        os.makedirs(output_folder, exist_ok=True)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.eval()
        model = torch.nn.Sequential(*list(model.children())[:-1])
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        features, positions = [], []
        def extract_features(patch):
            with torch.no_grad():
                return model(transform(patch).unsqueeze(0)).squeeze().numpy()
        for y in range(0, h - block_size + 1, stride):
            for x in range(0, w - block_size + 1, stride):
                patch = image[y:y+block_size, x:x+block_size]
                features.append(extract_features(patch))
                positions.append((x, y))
        features = np.array(features)
        similarity = cosine_similarity(features)
        detected = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                if similarity[i, j] > threshold:
                    detected.append((positions[i], positions[j]))
        for (x1, y1), (x2, y2) in detected:
            cv2.rectangle(image, (x1, y1), (x1 + block_size, y1 + block_size), (0, 255, 0), 2)
            cv2.rectangle(image, (x2, y2), (x2 + block_size, y2 + block_size), (0, 0, 255), 2)
        output_filename = os.path.splitext(os.path.basename(image_path))[0] + "_copymove.png"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, image)
        result["copy_move_image_path"] = f"copy_move_maps/{output_filename}"
        result["matched_regions_count"] = len(detected)
    except Exception as e:
        result["error"] = f"Copy-move forgery detection failed: {e}"
    return result


RAW_EXTS = {".cr2", ".nef", ".arw", ".orf", ".rw2", ".dng"}

def is_raw_file(path):
    return os.path.splitext(path)[1].lower() in RAW_EXTS

# ===== New Code Thumbnail/Camera identiication/CFA =====
# ===== Camera Model Identification =====
def identify_camera_model(image_path: str) -> dict:
    try:
        # Silence Python warnings/logging
        logging.getLogger("piexif").setLevel(logging.ERROR)
        logging.getLogger("exifread").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")

        make, model = None, None

        if is_raw_file(image_path):
            # Use exiftool for RAW
            try:
                cmd = [
                    "exiftool", "-s3", "-Make", "-Model", image_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                output = result.stdout.strip().split("\n")
                if len(output) >= 2:
                    make, model = output[0].strip(), output[1].strip()
            except Exception:
                pass
        else:
            # JPEG/TIFF path — pure Python
            try:
                # Pillow EXIF
                img = Image.open(image_path)
                pil_exif = img.getexif()
                if pil_exif:
                    for tag_id, val in pil_exif.items():
                        tag_name = Image.ExifTags.TAGS.get(tag_id, tag_id)
                        if tag_name == "Make" and not make:
                            make = str(val)
                        if tag_name == "Model" and not model:
                            model = str(val)
            except Exception:
                pass

            try:
                # exifread
                with open(image_path, "rb") as f:
                    tags = exifread.process_file(f, details=False)
                for key, val in tags.items():
                    if "Make" in key and not make:
                        make = str(val)
                    if "Model" in key and not model:
                        model = str(val)
            except Exception:
                return {"error": "Could not read camera information from RAW image."}

            try:
                # piexif
                exif_dict = piexif.load(image_path)
                zeroth = exif_dict.get("0th", {})
                if piexif.ImageIFD.Make in zeroth and not make:
                    make = zeroth[piexif.ImageIFD.Make].decode(errors="ignore")
                if piexif.ImageIFD.Model in zeroth and not model:
                    model = zeroth[piexif.ImageIFD.Model].decode(errors="ignore")
            except Exception:
                pass

        if not make and not model:
            return {"error": "Camera make and model could not be identified in this image."}
        return {"Make": make, "Model": model}

    except Exception as e:
        return {"error": str(e)}


# ===== Thumbnail Extraction & Comparison =====
def extract_thumbnail_and_compare(image_path: str) -> dict:
    try:
        # 1. Create media directory
        media_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "media"))
        os.makedirs(media_dir, exist_ok=True)

        # 2. Set the final thumbnail path inside media folder
        thumb_filename = os.path.basename(image_path).rsplit('.', 1)[0] + "_thumbnail.jpg"
        thumb_path = os.path.join(media_dir, thumb_filename)

        # 3. RAW file handling
        if is_raw_file(image_path):
            try:
                # ✅ Write to the correct media path
                with open(thumb_path, "wb") as out_file:
                    subprocess.run(
                        ["exiftool", "-b", "-PreviewImage", image_path],
                        stdout=out_file,
                        stderr=subprocess.DEVNULL
                    )
                if not os.path.exists(thumb_path) or os.path.getsize(thumb_path) == 0:
                    return {"error": "No thumbnail preview was found in this RAW image."}
            except Exception:
                return {"error": "Failed to extract thumbnail preview from this RAW image."}

        # 4. JPEG/TIFF handling
        else:
            exif_dict = piexif.load(image_path)
            if not exif_dict.get("thumbnail"):
                return {"error": "No thumbnail is embedded in this image’s metadata."}
            
            # ✅ Write thumbnail to correct media folder
            with open(thumb_path, "wb") as f:
                f.write(exif_dict["thumbnail"])

        # 5. Hash comparison
        thumb_hash = imagehash.phash(Image.open(thumb_path))
        main_hash = imagehash.phash(Image.open(image_path))
        similarity = 1 - (abs(thumb_hash - main_hash) / (len(main_hash.hash) ** 2))

        # 6. Return relative media path
        return {
            "thumbnail_path": f"/media/{thumb_filename}",
            "main_hash": str(main_hash),
            "thumbnail_hash": str(thumb_hash),
            "similarity_score": round(similarity, 4)
        }

    except Exception as e:
        return {"error": str(e)}


# ===== CFA Extraction =====
def detect_cfa_artifacts(image_path: str) -> dict:
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not read image"}

        # Extract green channel
        green_channel = img[:, :, 1].astype(np.float32)

        # Apply 2D Fourier Transform
        f_transform = np.fft.fft2(green_channel)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

        # Simple measure: high periodic peaks could mean CFA pattern present
        mean_val = np.mean(magnitude_spectrum)
        std_val = np.std(magnitude_spectrum)

        return {
            "mean_frequency_magnitude": float(mean_val),
            "std_frequency_magnitude": float(std_val),
            "possible_cfa_pattern":  bool(std_val > 50)  # heuristic threshold
        }
    except Exception as e:
        return {"error": str(e)}
    
# ======================= PRNU UTILITIES =======================
# Wavelet-denoise → noise residual → fingerprint (ML estimate)
def _to_gray_f32_bgr(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        g = img_bgr
    else:
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return (g.astype(np.float32) / 255.0)

def _wavelet_denoise(x: np.ndarray, wavelet="db2", level=4, sigma=None) -> np.ndarray:
    coeffs = pywt.wavedec2(x, wavelet, level=level)
    cA, cDs = coeffs[0], coeffs[1:]
    if sigma is None:
        HH = cDs[-1][2]
        sigma = np.median(np.abs(HH)) / 0.6745 + 1e-8

    new_cDs = []
    for (cH, cV, cD) in cDs:
        # BayesShrink-like soft threshold
        def soft(c):
            var = max(float((c*2).mean() - sigma*2), 0.0) + 1e-8
            thr = sigma**2 / (np.sqrt(var) + 1e-8)
            return np.sign(c) * np.maximum(np.abs(c) - thr, 0.0)
        new_cDs.append((soft(cH), soft(cV), soft(cD)))

    den = pywt.waverec2([cA] + new_cDs, wavelet)
    den = den[:x.shape[0], :x.shape[1]]
    return np.clip(den, 0.0, 1.0)

def prnu_noise_residual(img_bgr: np.ndarray) -> np.ndarray:
    g = _to_gray_f32_bgr(img_bgr)
    smooth = _wavelet_denoise(g)
    W = g - smooth
    W = W - cv2.GaussianBlur(W, (0, 0), 1.0)  # high-pass emphasis
    return W.astype(np.float32)

def _wiener_local(signal: np.ndarray, noise_var: float, win: int = 3) -> np.ndarray:
    mean = cv2.blur(signal, (win, win))
    mean_sq = cv2.blur(signal * signal, (win, win))
    var = np.maximum(mean_sq - mean * mean, 1e-8)
    return mean + (np.maximum(var - noise_var, 0.0) / (var + 1e-8)) * (signal - mean)

def prnu_estimate_fingerprint(images_bgr: List[np.ndarray]) -> np.ndarray:
    num = None
    den = None
    for img in images_bgr:
        g = _to_gray_f32_bgr(img)
        W = prnu_noise_residual(img)
        if num is None:
            num = (W * g)
            den = (g * g) + 1e-8
        else:
            num += (W * g)
            den += (g * g)
    K = num / (den + 1e-8)
    K = _wiener_local(K, noise_var=float(np.var(K)) * 0.5)
    return K.astype(np.float32)

def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    A = a - a.mean()
    B = b - b.mean()
    denom = (np.linalg.norm(A) * np.linalg.norm(B) + 1e-12)
    return float((A * B).sum() / denom)

def _cc_map(W: np.ndarray, K: np.ndarray, block: int = 64, stride: int = 32) -> np.ndarray:
    H, Wd = W.shape
    vals = []
    for y in range(0, H - block + 1, stride):
        row = []
        for x in range(0, Wd - block + 1, stride):
            ww = W[y:y+block, x:x+block]
            kk = K[y:y+block, x:x+block]
            row.append(_ncc(ww, kk))
        vals.append(row)
    return np.array(vals, dtype=np.float32)

def _pce(cc: np.ndarray, excl: int = 3) -> float:
    y, x = np.unravel_index(int(np.argmax(cc)), cc.shape)
    peak = cc[y, x]
    cc_bg = cc.copy()
    cc_bg[max(0, y-excl):min(cc.shape[0], y+excl+1),
          max(0, x-excl):min(cc.shape[1], x+excl+1)] = 0.0
    energy = float(np.mean(cc_bg**2) + 1e-12)
    return float((peak * peak) / energy)

def prnu_match_image_to_fingerprint(img_bgr: np.ndarray, K_ref: np.ndarray) -> Dict[str, float]:
    W = prnu_noise_residual(img_bgr)
    g = _to_gray_f32_bgr(img_bgr)
    Kh = K_ref[:g.shape[0], :g.shape[1]]
    K_proj = Kh * g
    ncc = _ncc(W, K_proj)
    cc = _cc_map(W, K_proj, block=64, stride=32)
    pce = _pce(cc)
    return {"ncc": ncc, "pce": pce}

def prnu_localize_single_image(img_bgr: np.ndarray, block: int = 64, stride: int = 16):
    g = _to_gray_f32_bgr(img_bgr)
    W = prnu_noise_residual(img_bgr)
    # Bias: self-fingerprint, good for inconsistency
    K_self = _wiener_local((W * g) / (g * g + 1e-8), noise_var=float(np.var(W)) * 0.5)
    cc = _cc_map(W, K_self, block=block, stride=stride)
    cc_norm = (cc - cc.min()) / (cc.max() - cc.min() + 1e-8)
    up = cv2.resize(cc_norm, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
    anomaly = 1.0 - up
    anomaly_u8 = np.clip(anomaly * 255.0, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(anomaly_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return up.astype(np.float32), mask
# ======================= /PRNU UTILITIES =======================

# ======================= PRNU ENDPOINTS =======================
#from fastapi import APIRouter

#prnu_router = APIRouter(prefix="/prnu", tags=["PRNU"])

#def _read_upload_to_bgr(f: UploadFile) -> np.ndarray:
    #data = f.file.read()
    #arr = np.frombuffer(data, np.uint8)
    #img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    #if img is None:
        #raise ValueError(f"Could not decode: {f.filename}")
    # Optional: downscale very large images for speed
    #h, w = img.shape[:2]
    #scale = 1280 / max(h, w)
    #if scale < 1.0:
        #img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    #return img

#@prnu_router.post("/fingerprint")
#async def build_fingerprint(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    #imgs = [_read_upload_to_bgr(f) for f in files]
    #K = prnu_estimate_fingerprint(imgs)
    # compact return for client-side storage (optional)
    #K_small = cv2.resize(K, (K.shape[1]//2, K.shape[0]//2), interpolation=cv2.INTER_AREA)
    #Kq = (np.clip((K_small - K_small.mean()) / (K_small.std() + 1e-8), -3, 3) * 8192).astype(np.int16)
    #checksum = int(np.abs(Kq).sum())
    # persist server-side if you want
    #np.save(os.path.join("prnu_maps", f"K_{checksum}.npy"), K)
    #return {"status": "ok", "shape": list(K.shape), "checksum": checksum, "saved_as": f"/prnu_maps/K_{checksum}.npy"}

#@prnu_router.post("/match")
#async def match_camera(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    #"""
    #Send at least 2 files:
    #- first N-1 images are reference (same camera)
    #- last image is the query
    #"""
    #if len(files) < 2:
        #return {"error": "Provide at least 2 images (N-1 refs, 1 query)."}
    #imgs = [_read_upload_to_bgr(f) for f in files]
    #K = prnu_estimate_fingerprint(imgs[:-1])
    #q = imgs[-1]
    #K = cv2.resize(K, (q.shape[1], q.shape[0]), interpolation=cv2.INTER_CUBIC)
    #scores = prnu_match_image_to_fingerprint(q, K)
    #is_match = (scores["ncc"] > 0.01 and scores["pce"] > 50.0)  # tune for your pipeline
    #return {"status": "ok", "ncc": scores["ncc"], "pce": scores["pce"], "match": bool(is_match)}

#@prnu_router.post("/localize")
#async def localize_prnu(file: UploadFile = File(...)) -> Dict[str, Any]:
    #img = _read_upload_to_bgr(file)
    #heat, mask = prnu_localize_single_image(img, block=64, stride=16)

    # Color heatmap and overlay
    #heat_u8 = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
    #heat_cmap = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    #overlay = cv2.addWeighted(img, 0.6, heat_cmap, 0.4, 0)

    # Save for static access
    #base = os.path.splitext(os.path.basename(file.filename or "upload"))[0]
    #heat_path = os.path.join("prnu_maps", f"{base}_prnu_heat.png")
    #mask_path = os.path.join("prnu_maps", f"{base}_prnu_mask.png")
    #prev_path = os.path.join("prnu_maps", f"{base}_prnu_overlay.png")
    #cv2.imwrite(heat_path, heat_cmap)
    #cv2.imwrite(mask_path, mask)
    #cv2.imwrite(prev_path, overlay)

    # Base64 (easy to show in frontend)
    #def _b64(p):
        #with open(p, "rb") as f:
            #return base64.b64encode(f.read()).decode("utf-8")

    #return {
        #"status": "ok",
        #"heatmap_png_b64": _b64(heat_path),
        #"mask_png_b64": _b64(mask_path),
        #"overlay_png_b64": _b64(prev_path),
        #"heatmap_path": f"/prnu/{os.path.basename(heat_path)}",
        #"mask_path": f"/prnu/{os.path.basename(mask_path)}",
        #"overlay_path": f"/prnu/{os.path.basename(prev_path)}",
    #}

#app.include_router(prnu_router)

mp_face_mesh = mp.solutions.face_mesh

def analyze_face_morphing(image_path: str) -> Dict[str, Any]:
    """Detect possible face morphing / swaps based on landmarks & blending artifacts"""
    results = {
        "face_detected": False,
        "morphing_suspected": False,
        "asymmetry_score": None,
        "blend_artifact_score": None,
        "error": None,
    }
    try:
        img = cv2.imread(image_path)
        if img is None:
            results["error"] = "Could not load image"
            return results

        h, w, _ = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        ) as face_mesh:
            output = face_mesh.process(rgb_img)

            if not output.multi_face_landmarks:
                results["error"] = "No face detected"
                return results

            results["face_detected"] = True
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in output.multi_face_landmarks[0].landmark]

            # --- Step 1: Asymmetry ---
            left_pts = [p for i, p in enumerate(landmarks) if i % 2 == 0]
            right_pts = [p for i, p in enumerate(landmarks) if i % 2 == 1]
            if left_pts and right_pts:
                left_center = np.mean(left_pts, axis=0)
                right_center = np.mean(right_pts, axis=0)
                asymmetry_score = float(np.linalg.norm(left_center - right_center))
                results["asymmetry_score"] = asymmetry_score
            else:
                results["asymmetry_score"] = 0.0

            # --- Step 2: Blending artifacts (sharpness variation) ---
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            results["blend_artifact_score"] = laplacian_var

            # --- Heuristic decision ---
            if results["asymmetry_score"] > 20 or results["blend_artifact_score"] < 100:
                results["morphing_suspected"] = True

    except Exception as e:
        results["error"] = str(e)

    return results

        
# ===== Emotion Analysis =====
def analyze_emotion(image_path: str):
    try:
        # First, try to detect faces with a high confidence threshold
        from deepface import DeepFace
        
        # Use a face detector first to check if there are any faces
        try:
            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend='opencv',
                enforce_detection=False
            )
            
            # If no faces or very low confidence, return appropriate message
            if not faces or len(faces) == 0:
                return {"error": "No faces were detected in the image. Please upload a photo with a visible face." }
                
            # Check the confidence of the best face detection
            best_face = max(faces, key=lambda x: x.get('confidence', 0))
            confidence = best_face.get('confidence', 0)
            
            if confidence < 0.7:  # High confidence threshold
                return {"error": "No faces were detected in the image. Please upload a photo with a visible face."}
                
        except Exception as face_detect_error:
            return f"Face detection failed: {str(face_detect_error)}"
        
        # If we get here, there are likely real faces, so proceed with emotion analysis
        emotion_analysis = DeepFace.analyze(
            img_path=image_path, 
            actions=['emotion'], 
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        if not emotion_analysis or len(emotion_analysis) == 0:
            return "Could not analyze emotions"
        
        result = emotion_analysis[0]
        dominant_emotion = result.get('dominant_emotion', 'unknown')
        emotion_scores = result.get('emotion', {})
        
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_scores": {k: round(v, 2) for k, v in emotion_scores.items()},
            "face_confidence": round(result.get('face_confidence', 0), 2),
            "face_region": result.get('region', {})
        }
        
    except Exception as e:
        return f"Emotion analysis failed: {str(e)}"

def detect_objects(image_path: str) -> list:
    """Run YOLOv8 object detection and return detected labels with confidence scores."""
    results = yolo_model.predict(image_path, verbose=False)  # disable console spam
    objects = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])  # class index
            label = yolo_model.names[cls_id]
            confidence = float(box.conf[0])
            objects.append({
                "label": label,
                "confidence": round(confidence, 2)
            })

    return objects

# ===== Endpoint =====
@app.post("/process-images")
async def process_uploaded_images(files: List[UploadFile] = File(...)):
    supported_formats = ('.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp')
    results = []
    imgs_bgr = []
    file_names = []

    save_dir = "images"
    os.makedirs(save_dir, exist_ok=True)

    for uploaded_file in files:
        if not uploaded_file.filename.lower().endswith(supported_formats):
            continue
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.filename)[1]) as temp_file:
            shutil.copyfileobj(uploaded_file.file, temp_file)
            temp_path = temp_file.name
            
        final_image_path = os.path.join(save_dir, uploaded_file.filename)
        shutil.copy(temp_path, final_image_path)
        #image_path = f"images/{uploaded_file.filename}"
        #file_names.append(uploaded_file.filename)

        # ========== Metadata & Forensics ==========
        pil_metadata = extract_pil_metadata(temp_path)
        exif_metadata = extract_exifread_metadata(temp_path)
        camera_model_info = identify_camera_model(temp_path)
        thumbnail_info = extract_thumbnail_and_compare(temp_path)
        cfa_info = detect_cfa_artifacts(temp_path)
        jpeg_structure_metadata = extract_jpeg_structure_metadata(temp_path)
        ela_image_path = perform_ela(temp_path)
        splicing_ela_path = generate_splicing_ela(temp_path)
        lighting_result = analyze_lighting_inconsistencies(temp_path)
        copy_move_result = detect_copy_move(temp_path)
        md5 = calculate_hash(temp_path, 'md5')
        sha256 = calculate_hash(temp_path, 'sha256')
        perceptual = perceptual_hash(temp_path)
        noise_image_path = visualize_noise(temp_path)
        noise_regions = region_noise_variation(temp_path)
        lighting_hist = lighting_histogram(temp_path)
        digest_info = extract_digest_info(temp_path)
        jpeg_quality_details = get_jpeg_quality_details(final_image_path)
        #======morphing
        face_morphing_result = analyze_face_morphing(temp_path)
        #======emotions
        emotion_result = analyze_emotion(temp_path)
        object_detection_result = detect_objects(temp_path)

        # ========== PRNU Localization ==========
        prnu_data = {}
        try:
            img_bgr = cv2.imread(temp_path, cv2.IMREAD_COLOR)
            imgs_bgr.append(img_bgr)
            if img_bgr is not None:
                heat, mask = prnu_localize_single_image(img_bgr, block=64, stride=16)
                heat_u8 = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
                heat_cmap = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(img_bgr, 0.6, heat_cmap, 0.4, 0)

                base = os.path.splitext(os.path.basename(uploaded_file.filename))[0]
                prnu_heat_path = os.path.join("prnu_maps", f"{base}_heat.png")
                prnu_mask_path = os.path.join("prnu_maps", f"{base}_mask.png")
                prnu_overlay_path = os.path.join("prnu_maps", f"{base}_overlay.png")
                cv2.imwrite(prnu_heat_path, heat_cmap)
                cv2.imwrite(prnu_mask_path, mask)
                cv2.imwrite(prnu_overlay_path, overlay)

                prnu_data = {
                    "localization": {
                        "heatmap_path": f"/prnu_maps/{os.path.basename(prnu_heat_path)}",
                        "mask_path": f"/prnu_maps/{os.path.basename(prnu_mask_path)}",
                        "overlay_path": f"/prnu_maps/{os.path.basename(prnu_overlay_path)}"
                    }
                }
            else:
                prnu_data = {"localization": {"error": "OpenCV could not read image for PRNU."}}
        except Exception as e:
            prnu_data = {"localization": {"error": f"PRNU localization failed: {e}"}}

        # ========== Merge Everything ==========
        results.append({
            "source_image_path": f"/images/{uploaded_file.filename}",
            #"image": uploaded_file.filename,
            "metadata_pil": pil_metadata,
            "metadata_exifread": exif_metadata,
            "ela_image_path": ela_image_path,
            "lighting_inconsistencies": lighting_result,
            "hashes": {
                "md5": md5,
                "sha256": sha256,
                "perceptual": perceptual
            },
            "noise_analysis": {
                "noise_map_path": noise_image_path,
                "regional_variation": noise_regions
            },
            "lighting_histogram": lighting_hist,
            "copy_move_forgery": {
                "map_path": copy_move_result.get("copy_move_image_path", ""),
                "matches_found": copy_move_result.get("matched_regions_count", 0),
                "error": copy_move_result.get("error", "")
            },
            "splicing_analysis": {
                "ela_splicing_image": splicing_ela_path
            },
            "jpeg_structure_metadata": jpeg_structure_metadata,
            "digest_info": digest_info,
            "jpeg_quality_details": jpeg_quality_details,
            "camera_model": camera_model_info,
            "thumbnail_analysis": thumbnail_info,
            "cfa_analysis": cfa_info,
             # ===== New Code 25/Aug/25 =====
             
            "emotion_analysis": emotion_result,
            "face_region_analysis": face_morphing_result,
            "object_detection": object_detection_result,
            "prnu": prnu_data   # <== merged directly here
        })

    # ========== PRNU Fingerprint & Match ==========
    try:
        if len(imgs_bgr) >= 1:
            K = prnu_estimate_fingerprint(imgs_bgr)
            checksum = int(np.abs(K).sum())
            np.save(os.path.join("prnu_maps", f"K_{checksum}.npy"), K)
            for r in results:
                r["prnu"]["fingerprint"] = {
                    "status": "ok",
                    "shape": list(K.shape),
                    "checksum": checksum,
                    "saved_as": f"/prnu_maps/K_{checksum}.npy"
                }

        if len(imgs_bgr) >= 2:
            K_ref = prnu_estimate_fingerprint(imgs_bgr[:-1])
            q = imgs_bgr[-1]
            K_ref = cv2.resize(K_ref, (q.shape[1], q.shape[0]), interpolation=cv2.INTER_CUBIC)
            scores = prnu_match_image_to_fingerprint(q, K_ref)
            is_match = (scores["ncc"] > 0.01 and scores["pce"] > 50.0)
            results[-1]["prnu"]["match"] = {
                "status": "ok",
                "ncc": scores["ncc"],
                "pce": scores["pce"],
                "match": bool(is_match),
                "reference_images": file_names[:-1]
            }
    except Exception as e:
        for r in results:
            r["prnu"]["fingerprint"] = {"error": f"PRNU fingerprint failed: {e}"}
            r["prnu"]["match"] = {"error": f"PRNU match failed: {e}"}

    return {"results": results}

from datetime import datetime

# === Dashboard Endpoints ===
@app.get("/recent-uploads")
def get_recent_uploads(limit: int = 5):
    images_folder = "images"
    if not os.path.exists(images_folder):
        return {"recent_uploads": []}

    files_with_time = []
    for filename in os.listdir(images_folder):
        file_path = os.path.join(images_folder, filename)
        if os.path.isfile(file_path):
            mtime = os.path.getmtime(file_path)
            files_with_time.append({
                "filename": filename,
                "url": f"/images/{filename}",
                "timestamp": datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            })

    files_with_time.sort(
        key=lambda x: os.path.getmtime(os.path.join(images_folder, x["filename"])),
        reverse=True
    )
    return {"recent_uploads": files_with_time[:limit]}


@app.get("/analyzed-results")
def get_analyzed_results():
    def list_files(folder, filter_func=None):
        if not os.path.exists(folder):
            return []
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        if filter_func:
            files = [f for f in files if filter_func(f)]
        return files

    data = {
        "total_images": list_files("images"),
        "ela_images": list_files("ela_results", lambda f: "_ela" in f.lower()),
        "splicing_images": list_files("ela_results", lambda f: "splicing" in f.lower()),
        "copy_move_images": list_files("copy_move_maps"),
        "noise_map_images": list_files(temp_dir, lambda f: f.endswith("_noise.png")),
        "lighting_heatmaps": list_files("lighting_maps"),
        "fingerprints": list_files("prnu_maps", lambda f: f.endswith(".png")),
    }

    stats = {key: len(files) for key, files in data.items()}
    return {"stats": stats, "files": data}
