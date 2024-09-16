ffmpeg -framerate 5 -pattern_type glob -i '*-0.png' progress.gif
ffmpeg -i progress.gif -filter_complex "[0:v]crop=512:512:0:0[out1]; [0:v]crop=512:512:512:0[out2]; [0:v]crop=512:512:1024:0[out3]" -map "[out1]" progress_rgb.gif -map "[out2]" progress_geo.gif -map "[out3]" progress_msk.gif
ffmpeg -i it20000-test.mp4 -vf "fps=10,scale=1536:-1:flags=lanczos" -c:v gif rotate.gif
ffmpeg -i rotate.gif -filter_complex "[0:v]crop=512:512:0:0[out1]; [0:v]crop=512:512:512:0[out2]; [0:v]crop=512:512:1024:0[out3]" -map "[out1]" rotate_rgb.gif -map "[out2]" rotate_geo.gif -map "[out3]" rotate_msk.gif
