# OCR Image Generator
This code is used to generate some synthesized text pictures to train the text recognition model.

## Prepare ENV

```
docker build -t ocr_generator_image .
docker run -itd -v $PWD:/mnt --name ocr_generator ocr_generator_image
```

## Examples of generating images

| parameter  |    Example images 1    |      Example images 2 |        Example images 3 |
| ---         |     ---      |          --- |           --- |
| `--blur `| <img src="./demo_image/img_3_blur.jpg" width="200" height="32" >    |   <img src="./demo_image/img_3_blur2.jpg" width="200" height="32">    |  <img src="./demo_image/img_3_blur46.jpg" width="200" height="32">   |
| `--prydown`| <img src="./demo_image/mi1.jpg" width="200" height="32">      |    <img src="./demo_image/mi2.jpg" width="200" height="32">    |   <img src="./demo_image/mi3.jpg" width="200" height="32">  
| `--lr_motion`| <img src="./demo_image/lf1.jpg" width="200" height="32">  |   <img src="./demo_image/lf2.jpg" width="200" height="32">   |  <img src="./demo_image/lf4.jpg" width="200" height="32"> 
| `--ud_motion`| <img src="./demo_image/img_3_up2.jpg" width="200" height="32">      |    <img src="./demo_image/img_3_up5.jpg" width="200" height="32">    |   <img src="./demo_image/img_3_up22.jpg" width="200" height="32">    |
|`--random_offset` | <img src="./demo_image/rd1.jpg" width="50" height="40">    |   <img src="./demo_image/rd2.jpg" width="60" height="45">    |  <img src="./demo_image/rd3.jpg" width="160" height="60">  
|`noise_enable` | <img src="./demo_image/n1.jpg" width="200" height="32">    |   <img src="./demo_image/n2.jpg" width="200" height="32">    |  <img src="./demo_image/n3.jpg" width="200" height="32">  

## Random spaces about generating text
Because the text in the real scene is likely to have a certain gap, if the distance of the text you generate is fixed,
it is likely that the text in the trained model cannot be recognized with too large a gap. This code solves this very well.
This code solves this problem by adding random spaces to the code. The specific effects are as follows:<br>
|   Example images 1    |      Example images 2 |        Example images 3 |
| ---      |          --- |           --- |
| <img src="./demo_image/img_3_space15.jpg" width="250">    |   <img src="./demo_image/img_3_space57.jpg" width="250">    |  <img src="./demo_image/img_3_space79.jpg" width="250">   |

## Arguments

```
usage: main.py [-h] [--num_img NUM_IMG] [--font_min_size FONT_MIN_SIZE]
               [--font_max_size FONT_MAX_SIZE] [--bg_path BG_PATH]
               [--fonts_path FONTS_PATH] [--corpus_path CORPUS_PATH]
               [--chars_file CHARS_FILE] [--blur] [--prydown] [--lr_motion]
               [--ud_motion] [--random_offset] [--config_file CONFIG_FILE]
               [--random_augmentation] [--output_dir OUTPUT_DIR]
               [--label_file LABEL_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --num_img NUM_IMG     Number of images per text to generate
  --font_min_size FONT_MIN_SIZE
                        Can help adjust the size of the generated text and the
                        size of the picture
  --font_max_size FONT_MAX_SIZE
                        Can help adjust the size of the generated text and the
                        size of the picture
  --bg_path BG_PATH     The generated text pictures will use the pictures of
                        this folder as the background
  --fonts_path FONTS_PATH
                        The font used to generate the picture
  --corpus_path CORPUS_PATH
                        The corpus used to generate the text picture
  --chars_file CHARS_FILE
                        Chars allowed to be appear in generated images
  --blur                Apply gauss blur to the generated image
  --prydown             Blurred image, simulating the effect of enlargement of
                        small pictures
  --lr_motion           Apply left and right motion blur
  --ud_motion           Apply up and down motion blur
  --random_offset       Randomly add offset
  --config_file CONFIG_FILE
                        Set the parameters when rendering images
  --random_augmentation
                        Set random augmentations
  --output_dir OUTPUT_DIR
                        Images save dir
  --label_file LABEL_FILE
                        Label txt save path
```

## Reference
- https://github.com/clovaai/synthtiger
- https://github.com/zcswdt/Color_OCR_image_generator