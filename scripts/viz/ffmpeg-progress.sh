ffmpeg -framerate 10 -pattern_type glob -i '*.png' progress.gif
ffmpeg -i progress.gif -filter_complex "[0:v]crop=512:512:0:0[out1]; [0:v]crop=512:512:512:0[out2]; [0:v]crop=512:512:1024:0[out3]" -map "[out1]" progress_rgb.gif -map "[out2]" progress_geo.gif -map "[out3]" progress_msk.gif
ffmpeg -i it15000-test.mp4 -vf "fps=10,scale=1536:-1:flags=lanczos" -c:v gif rotate.gif
ffmpeg -i rotate.gif -filter_complex "[0:v]crop=512:512:0:0[out1]; [0:v]crop=512:512:512:0[out2]; [0:v]crop=512:512:1024:0[out3]" -map "[out1]" rotate_rgb.gif -map "[out2]" rotate_geo.gif -map "[out3]" rotate_msk.gif
ffmpeg -i rotate.gif -filter_complex "[0:v]crop=512:512:0:0[out1]; [0:v]crop=512:512:512:0[out2]" -map "[out1]" rotate_rgb.gif -map "[out2]" rotate_msk.gif
ffmpeg -i progress.gif -filter_complex "[0:v]crop=512:512:0:0[out1]; [0:v]crop=512:512:512:0[out2]; [0:v]crop=512:512:1024:0[out3]; [0:v]crop=512:512:1536:0[out4]; [0:v]crop=512:512:2048:0[out5]" -map "[out1]" progress_rgb.gif -map "[out2]" progress_geo.gif -map "[out3]" progress_cls_geo.gif -map "[out4]" progress_cls_rgb.gif -map "[out5]" progress_msk.gif
