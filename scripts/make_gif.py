import glob
from PIL import Image

# filepaths
fp_in = "../output/pie_chart/pie_*.png"
fp_out = "../output/pie_chart/pie.gif"

files = glob.glob(fp_in)
files = [f.split('_')[2].split('.')[0] for f in files]
files = sorted(files, key=int)

img, *imgs = [Image.open(f'../output/pie_chart/pie_{f}.png') for f in files]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=500, loop=0)
