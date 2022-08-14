# ANIFY
Anify is a program to 2d animate a teacher into an animated character in an online meet.
This could also be used in any online meet to bring fun into their environment

## Demo


## Features
- Currently developed program to animate the person into Shinchan
- Sensitive to blinking eye, opening mouth, moving hands, legs, body and head.
- Mimic function is used to change the pitch of the voice, to bring in cartoonic voices, which in turn engages student

> Note : This is a basic animation software. It isn't fully functional and has many distortion.
> > Recommended to not to move while using program to smooth animation.

## Steps to run the program
- Clone the repository
- Install the required libraries
- Turn on the webcam
- Run the main.py
- Open the final.svg in a live server(Eg. browser). Reload the page continously to see seamless animation
- Uncomment the below line in main.py to add the voice effect.
        # Thread(target = mimic).start()
    It is recommended to use the mimic function in noise less environment for better experience.


## Libraries

Anify uses a different open source and in built libraries :

- opencv (To get video input from webcam)
- mediapipe (To get the facial and body landmark coordinates)
- numpy 
- xml.dom.minidom (To manipulate the SVG image)
- pyaudio (To change teh pitch of the voice)

## Installation

Clone the project and install necessary packages with the below code (Windows)

```bash
  pip install -r requirements.txt
```
  
 And then run main.py file.

## Contribution

* [Kasinath J](https://www.linkedin.com/in/kasinath-j-2881a6200/)
* [Sorna Sarathi N](https://www.linkedin.com/in/sorna-sarathi-9b2a54240/)
* [Rohith N M](https://www.linkedin.com/in/rohith-n-m-087a09229/)


## License
MIT

