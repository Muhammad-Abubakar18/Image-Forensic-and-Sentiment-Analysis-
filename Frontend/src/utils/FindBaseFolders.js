export default function findBaseFolder(filename) {
  if (filename.includes("_ela")) return "ela_results";
  if (filename.includes("splicing")) return "ela_results";
  if (filename.includes("_noise")) return "tmp";
  if (filename.includes("copymove")) return "copy_move_maps";
  if (filename.includes("heatmap")) return "lighting_maps";
  return "images";
}
