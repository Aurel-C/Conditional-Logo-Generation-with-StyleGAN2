# Conditional Logo Generation with StyleGAN2

This project is an application of StylGAN2 for logo generation with Tensorflow and Flask.

The model implementation comes from [here](https://github.com/manicman1999/StyleGAN2-Tensorflow-2.0/) with some changes to support conditional GANs and run on CPU.

Original paper: [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958)

Run the program with

```bash
$ python app.py
```

## Example of randomly generated logos
![Teaser image](./assets/example.png)

## Variation of the same logo by changing condition and style
![Image with condition and style](./assets/example-cond-style.png)

## Screenshot of GUI
![GUI](./assets/example-GUI.png)
