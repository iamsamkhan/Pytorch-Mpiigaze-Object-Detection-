Install requirements.txt in pytorch_mpiigaze_demo parent directory

Change the variables "video_path" and "output_dir" declared in the main function of main.py file 
to set paths of input video file and directory to store generated csv file respectively

Run __main__.py to run the codes

The specific lines where csv file is stored are in demo.py lines 251 and 355

The code behaviour is as follows:
Case: Empty Video file
  An empty csv file is stored and value error "Empty Video" is raised.
Case: No faces detected in any frame
  An empty csv file is stored and value error "No Face Detected" is raised.
Case: At least one face is detected in at least one frame
  A csv file is stored along with data.
