# turn set of PNGs into a GIF
png_dir="/home/ubuntu/Desktop/to_animate"
gif_dir="/home/ubuntu/Desktop/to_animate"
cd ${png_dir}
for size in 1568; do
ffmpeg -i final_fps24_size${size}.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" final_fps24_size${size}.mp4
done