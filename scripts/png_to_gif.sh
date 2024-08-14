# turn set of PNGs into a GIF
png_dir="/home/ubuntu/Desktop/to_animate"
gif_dir="/home/ubuntu/Desktop/to_animate"
for fps in 24; do
for x_size in 1568; do
ffmpeg -pattern_type glob -i "${png_dir}/*.png" -filter_complex "setpts=1*PTS,scale=${x_size}:-1,split=2[s0][s1];[s0]palettegen[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5" ${gif_dir}/final_fps${fps}_size${x_size}.gif
done
done