ffmpeg -i it15000-test.mp4 -vf "fps=10,scale=1536:-1:flags=lanczos" -c:v gif rotate.gif
ffmpeg -i it15000-test.mp4 -vf "fps=10,scale=2560:-1:flags=lanczos" -c:v gif rotate.gif
ffmpeg -i rotate.gif -filter_complex "[0:v]crop=512:512:0:0[out1]; [0:v]crop=512:512:512:0[out2]; [0:v]crop=512:512:1024:0[out3]" -map "[out1]" rotate_rgb.gif -map "[out2]" rotate_geo.gif -map "[out3]" rotate_msk.gif
ffmpeg -i rotate.gif -filter_complex "[0:v]crop=512:512:0:0[out1]; [0:v]crop=512:512:512:0[out2]; [0:v]crop=512:512:1024:0[out3]" -map "[out1]" rotate_rgb.gif -map "[out2]" rotate_cls.gif -map "[out3]" rotate_msk.gif
ffmpeg -i rotate.gif -filter_complex "[0:v]crop=512:512:0:0[out1]; [0:v]crop=512:512:512:0[out2]; [0:v]crop=512:512:1024:0[out3]; [0:v]crop=512:512:1536:0[out4]; [0:v]crop=512:512:2048:0[out5]" -map "[out1]" rotate_rgb.gif -map "[out2]" rotate_geo.gif -map "[out3]" rotate_clothes_normal.gif -map "[out4]" rotate_clothes.gif -map "[out5]" rotate_mask.gif
