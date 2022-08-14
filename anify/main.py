from animate import animate
from mimic import mimic
from threading import Thread

if __name__ == '__main__':
    Thread(target = animate).start()
    Thread(target = mimic(300)).start()

# Open the final.svg in a browser an reload it if needed to see the seamless animation
# Use the mimic-voice changer in a very silent environment for good experience, if not needed comment it

