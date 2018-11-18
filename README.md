# BoostingSpeech
1) Copy the test folder from TEDLIUM into this path.
  BoostingSpeech/TEDLIUM_release1/test
2) sph directory - .sph and .wav files
3) stm directory - .stm files and output.txt (the stm file converted into int sequence where each alphabet, space, underscore and apostrphe is mapped to a number, the mapping is available in preprocessing_text.py)



## Instructions
sudo apt-get install sox
cd to .sph directory
The following command will convert .sph to .wav and store it in the same directory 
for f in *.sph; do sox -t sph "$f" -b 16  -t wav "${f%.*}.wav"; done

we may have to do this and then upload to TACC
