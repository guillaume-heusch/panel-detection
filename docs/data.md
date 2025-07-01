# THE DATA

There are 29 video sequences, but I don't know with which device they have been captured. They all vary in length and overall quality I'd say. They are quite diverse, in the the sense that sometimes it's taken from the back of the venue, sometimes from the stage, and sometimes from the side. That makes a good set of different examples though.

## Extracting frames

Extracting individual frames from each sequence has been done with the simple ffmpeg command:

```shell
ffmpeg -i {video_file} {dir_with_frames}/{sequence_name}-frame%04d.png
```

**WARNING:** The `{dir_with_frames}` should exist before running the command.

**There is a grand total of 14932 frames**, with quite some redundancy (which is normal in a video sequence). I deliberately not chose to eliminate some, but I'm quite aware that this dataset is somehow not optimal because of that.

## Getting annotations

For each frame, an annotation file has been generated as a csv file. Each csv file contains one or several lines, i.e. one line for each panel correctly detected. By correctly detected, I mean that the bounding box around the panel is more or less ok, and that the correct number has been detected by the OCR engine. 

Here's an example of the content of such a csv file:

```
20,974,833,1075,899
861,2987,986,3143,1088
```

The first number is the detected number on the panel and the four remaining numbers are the coordinates of the bounding box, as [xmin, ymin, xmax, ymax].

These detections have been automatically made using a [fine-tuned Faster R-CNN](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html) for the bounding box detection and with the [OCR Tamil package](https://github.com/gnana70/tamil_ocr) for recognizing the numbers inside each panel.

Faulty annotations (where the number is not correctly recognized) have not been corrected, but simply been removed from the annotation file - This would have required way too much effort. Note that this may cause issues for future evaluations, since the ground truth is *de facto* not exact. 

### Some statistics

In some images, no panels can be detected. Consequently no csv file is generated for such images. This can happen for various reasons (an obvious one is that there is actually no panels in the image). Here's an example of a blurry image when no panels are detected:

![](img/blurry_example.jpg)  

**Of the 14932 images in total, there are XXX in which no panels are detected.**