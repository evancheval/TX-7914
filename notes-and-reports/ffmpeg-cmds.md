# Modify video 
TO SPEED UP A VIDEO BY 2X
```bash
ffmpeg -i input.mp4 -filter:v "setpts=0.5*PTS" -an output.mp4
```

to get a video with one frame per second
```bash
ffmpeg -i input.mp4 -vf fps=1 output.mp4
```

to get a video with one frame per second between frame 600 and 1200
```bash
ffmpeg -i data/TD_DIO5_Seance2_Box4_Groupe1_Part1_Up_left.mp4 -vf "fps=1, select='between(n,600,1200)'" -vsync 0 -an data/ffmpeg_output/output.mp4
```

ALTERNATIVE : UTILISER LES TIMESTAMPS (PLUS SIMPLE)
```bash
ffmpeg -i data/TD_DIO5_Seance2_Box4_Groupe1_Part1_Up_left.mp4 -ss 00:12:00 -to 00:20:00 -vf fps=5 -an data/ffmpeg_output/output.mp4
```

# Extracting frames from a video using ffmpeg

TO GET ONE FRAME PER SECOND
```bash
ffmpeg -i input.mp4 -vf fps=1 data/ffmpeg_output/frames/frame_%03d.png
```

TO EXTRACT FRAMES BETWEEN FRAME 18000 AND 30000 with 1/10 fps
```bash
rm data/ffmpeg_output/frames/frame_*.png
ffmpeg -i data/TD_DIO5_Seance2_Box4_Groupe1_Part1_Up_left.mp4 -vf "select='between(n,18000,30000)'" -an data/ffmpeg_output/frames/temp.mp4
ffmpeg -i data/ffmpeg_output/frames/temp.mp4 -vf "fps=0.5" data/ffmpeg_output/frames/frame_%03d.png
rm data/ffmpeg_output/frames/temp.mp4
```

