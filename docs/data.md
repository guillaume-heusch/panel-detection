# THE DATA

There are 29 video sequences, but I don't know with which device they have been captured. They all vary in length and overall quality I'd say.

They are quite diverse, in the the sense that sometimes it's taken from the back of the venue, sometimes from the stage, and sometimes from the side. That makes a good set of different examples though.

## Extracting frames

Extracting individual frames from each sequence has been done with the simple ffmpeg command:

```shell
ffmpeg -i {video_file} {dir_with_frames}/{sequence_name}-frame%04d.png
```

**WARNING:** The `{dir_with_frames}` should exist before running the command.

There is a grand total of 14932 frames, with quite some redundancy (which is normal in a video sequence). I deliberately not chose to eliminate some, but I'm quite aware that this dataset is somehow not optimal because of that.

## Getting annotations

For each frame, an annotation file has been generated as a csv file. 