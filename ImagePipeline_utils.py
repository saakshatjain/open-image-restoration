import time
import numpy as np
import os, sys, shutil
from contextlib import contextmanager
from numba import cuda as ncuda
import PIL
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import cv2
import contextlib
from copy import deepcopy
import subprocess
from glob import glob
from os import path as osp
from os import path
	
utilspath = os.path.join(os.getcwd(), 'utils/')

@contextmanager
def timing(description: str) -> None:
  
	start = time.time()
	
	yield
	elapsed_time = time.time() - start

	print( description + ': finished in ' + f"{elapsed_time:.4f}" + ' s' )


class Quiet:
	
	def __init__(self):
		
		#Store initial stdout in this variable
		self._stdout = sys.stdout
		
	def __del__(self):
		
		sys.stdout = self._stdout

	@contextmanager
	def suppress_stdout(self, raising = False):

		with open(os.devnull, "w") as devnull:
			error_raised = False
			error = "there was an error"
			sys.stdout = devnull
			try:  
				yield
			except Exception as e:
				error_raised = True  
				error = e
				sys.stdout = self._stdout
				print(e)
			finally:
				finished = True
				sys.stdout = self._stdout

		sys.stdout = self._stdout		 
		if error_raised:
			if raising:
				raise(error)
			else:
				print(error)


	#Mute stdout inside this context
	@contextmanager
	def quiet_and_timeit(self, description = "Process running", raising = False, quiet = True):

		print(description+"...", end = '')
		start = time.time()
		try:

			if quiet:
				#with suppress_stdout(raising):	
				sys.stdout = open(os.devnull, "w")
			yield
			if quiet:
				sys.stdout = self._stdout
		except Exception as e:
			if quiet:
				sys.stdout = self._stdout
			if raising:
				sys.stdout = self._stdout
				raise(e)
			else:
				sys.stdout = self._stdout
				print(e)

		elapsed_time = time.time() - start
		
		sys.stdout = self._stdout
		print(': finished in ' + f"{elapsed_time:.4f}" + ' s' )
		
		

	#Force printing in stdout, regardless of the context (such as the one defined above)	
	def force_print(self, value):
		prev_stdout = sys.stdout
		sys.stdout = self._stdout
		print(value)
		sys.stdout = prev_stdout



def duplicatedir(src,dst):
	
	if not os.path.exists(src):
		print('ImagePipeline_utils. duplicatedir: Source directory does not exists!')
		return
	
	if src != dst:
		
		if os.path.exists(dst):
			shutil.rmtree(dst)

		shutil.copytree(src=src,dst=dst) 

def createdir_ifnotexists(directory):
	#create directory, recursively if needed, and do nothing if directory already exists
	os.makedirs(directory, exist_ok=True)

def initdir(directory):

	if os.path.exists(directory):
		shutil.rmtree(directory)   
	os.makedirs(directory)
			
def to_RGB(image):
	return image.convert('RGB')

def to_grayscale(image):	
	return image.convert('L')

def split_RGB_images(input_dir):
	
	imname = '*'
	orignames = glob(os.path.join(input_dir, imname))
	
	for orig in orignames:
		
		try:
			im = Image.open(orig)			

			#remove alpha component
			im = to_RGB(im)
			
			#split channels
			r, g, b = Image.Image.split(im)
			r = to_RGB(r)
			g = to_RGB(g)
			b = to_RGB(b)


			#save as png (and remove previous version)
			f, e = os.path.splitext(orig)
			os.remove(orig)
			
			r.save(f+"_red.png")
			g.save(f+"_green.png")
			b.save(f+"_blue.png")
			
		except Exception as e:
			print(e)	

def unsplit_RGB_images(input_dir):
	
	imname = '*_red.png'
	orignames = glob(os.path.join(input_dir, imname))
	
	for orig in orignames:
		
		try:
			substring = orig[:-8]
			r = to_grayscale(Image.open(substring+'_red.png'))
			g = to_grayscale(Image.open(substring+'_green.png'))
			b = to_grayscale(Image.open(substring+'_blue.png'))
			
			im = Image.merge('RGB', (r,g,b) )
			
			#save as png (and remove monochannel images)
			os.remove(substring+'_red.png')
			os.remove(substring+'_green.png')
			os.remove(substring+'_blue.png')
			
			im.save(substring+".png")
			
		except Exception as e:
			print(e)			
			
	
			
def preprocess(input_dir, gray = True, resize = True, size = (1000,1000)):

	imname = '*'
	orignames = glob(os.path.join(input_dir, imname))
	
	for orig in orignames:
		
		try:
			im = Image.open(orig)

			#remove alpha component
			im = to_RGB(im)

			#convert to grayscale
			if gray:
				im = to_grayscale(im)

			#resize
			if resize:

				width, height = im.size

				#resize only if larger than limit
				if width > size[0] or height > size[1]:
					im.thumbnail(size,Image.ANTIALIAS)

			#save as png (and remove previous version)
			f, e = os.path.splitext(orig)
			os.remove(orig)
			im.save(f+".png")
		except Exception as e:
			print(e)

def filtering(input_dir, median = True, median_winsize = 5, mean = True, mean_winsize = 5):

	with timing("Filtering (median) with PIL (consider using filtering_opencv for faster processing)"):
		imname = '*'
		orignames = glob(os.path.join(input_dir, imname))

		for orig in orignames:

			try:
				im = Image.open(orig)

				
				#median blur
				if median:
					im = im.filter(ImageFilter.MedianFilter(median_winsize))  
					
				#mean blur
				if mean:
					im = im.filter(ImageFilter.Meanfilter(mean_winsize))				 

				#save as png (and remove previous version)
				f, e = os.path.splitext(orig)
				os.remove(orig)
				im.save(f+".png")
			except Exception as e:
				print(e)

def filtering_opencv(input_dir, median = True, median_winsize = 5, gaussian = True, gaussian_x = 5, gaussian_y = 5, gaussian_std = 0, mean = True, mean_winsize = 3):

	with timing("Filtering (median) with opencv"):
		imname = '*'
		orignames = glob(os.path.join(input_dir, imname))

		for orig in orignames:
			print(orig)
			try:
				im = cv2.imread(orig, cv2.IMREAD_COLOR)


				#median blur
				if median:
					im = cv2.medianBlur(im,median_winsize)	 
					
				if gaussian:
					im = cv2.GaussianBlur(im,(gaussian_x,gaussian_y),gaussian_std)

				#mean blur
				if mean:
					im = cv2.blur(im,(mean_winsize,mean_winsize))
					
				

				#save as png (and remove previous version)
				f, e = os.path.splitext(orig)
				os.remove(orig)
				cv2.imwrite(f+".png", im)
			except Exception as e:
				print(e)
	
			
def rotate_images(input_dir):

	imname = '*'
	orignames = glob(os.path.join(input_dir, imname))
	
	for orig in orignames:
		
		try:
			im = Image.open(orig)

			#remove alpha component
			im = im.transpose(Image.ROTATE_90)

			#save as png (and remove previous version)
			f, e = os.path.splitext(orig)
			os.remove(orig)
			im.save(f+".png")
		except Exception as e:
			print(e)

def unrotate_images(input_dir):

	imname = '*'
	orignames = glob(os.path.join(input_dir, imname))
	
	for orig in orignames:
		
		try:
			im = Image.open(orig)

			#remove alpha component
			im = im.transpose(Image.ROTATE_270)

			#save as png (and remove previous version)
			f, e = os.path.splitext(orig)
			os.remove(orig)
			im.save(f+".png")
		except Exception as e:
			print(e)			
			
def reset_gpu(device = 0):  
	
	ncuda.select_device(device)
	ncuda.close()
	
import os, time, datetime
#import PIL.Image as Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imread, imsave
from PIL import Image, ImageDraw, ImageFont
import shutil
import subprocess
from glob import glob
from copy import deepcopy

def to_tensor(img):
    if img.ndim == 2:
        return img[np.newaxis, ..., np.newaxis]
    elif img.ndim == 3:
        return np.moveaxis(img, 2, 0)[..., np.newaxis]

def from_tensor(img):
    return np.squeeze(np.moveaxis(img[..., 0], 0, -1))

def save_result(result, path):
    path = path if path.find('.') != -1 else path + '.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))

utilspath = os.path.join(os.getcwd(), 'utils/')
fontfile = os.path.join(utilspath, "arial.ttf")

def addnoise(im, sigma=10, imagetype='L', add_label=False):
    x = np.array(im)
    y = x + np.random.normal(0, sigma, x.shape)
    y = np.clip(y, 0, 255)
    im = Image.fromarray(y.astype('uint8'), imagetype)

    if add_label:
        d = ImageDraw.Draw(im)
        fnt = ImageFont.truetype(fontfile, 40)
        fill = 240 if imagetype == 'L' else (255, 0, 0) if imagetype == 'RGB' else (255, 0, 0, 0)
        d.text((10, 10), f"sigma = {sigma}", font=fnt, fill=fill)

    return im

def concat_images(img_list, labels=[], imagetype=None, sameheight=True, imagewidth=None, imageheight=None, labelsize=30, labelpos=(10, 10), labelcolor=None):
    images = deepcopy(img_list)
    imagetype = imagetype or 'RGB'
    images = [im.convert(imagetype) for im in images]
    
    if imageheight is not None:
        sameheight = True

    widths, heights = zip(*(i.size for i in images))
    if ((len(set(heights)) > 1) & sameheight) or (imageheight is not None) or (imagewidth is not None):
        imageheight = imageheight or min(heights)
        if imagewidth is not None:
            images = [im.resize((int(imagewidth), int(imageheight)), Image.ANTIALIAS) if sameheight else
                      im.resize((int(imagewidth), int(im.height * imagewidth / im.width)), Image.ANTIALIAS) for im in images]
        else:
            images = [im.resize((int(im.width * imageheight / im.height), imageheight), Image.ANTIALIAS) for im in images]

    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new(imagetype, (total_width, max_height))

    if len(labels) == len(images):
        fnt = ImageFont.truetype(fontfile, labelsize)
        fill = 240 if imagetype == 'L' else (176, 196, 222) if imagetype == 'RGB' else (176, 196, 222, 0)
        fill = labelcolor or fill
        for i in range(len(labels)):
            ImageDraw.Draw(images[i]).text(labelpos, labels[i], font=fnt, fill=fill)

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return new_im

def display_images(im_list, labels=[], **kwargs):
    display(concat_images(im_list, labels, **kwargs))

def get_filepaths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

def get_filenames(directory):
    return [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

def display_folder(directory, limit=10, **kwargs):
    files = get_filepaths(directory)
    files.sort()
    files = files[:limit] if len(files) > limit else files
    display_images([Image.open(f) for f in files], [os.path.split(f)[1] for f in files], **kwargs)

def compare_folders(dirs, labels=[], **kwargs):
    dirlist = dirs if isinstance(dirs, list) else [d for d in glob(os.path.join(dirs, '*')) if os.path.isdir(d)]
    names = sorted(get_filenames(dirlist[0]))
    for n in names:
        paths = [glob(os.path.join(d, os.path.splitext(n)[0] + '*'))[0] for d in dirlist]
        display_images([Image.open(p) for p in paths], [os.path.split(d)[1] for d in dirlist], **kwargs)

def clone_git(url, dir_name=None, tag=None, reclone=False):
    old_dir = os.getcwd()
    dir_name = dir_name or os.path.splitext(os.path.split(url)[1])[0]
    
    if reclone and os.path.exists(dir_name):
        shutil.rmtree(dir_name)
        
    if not os.path.exists(dir_name):
        subprocess.run(f"git clone {url} {dir_name}", shell=True)
        
    os.chdir(dir_name)
    if tag is not None:
        subprocess.run(f"git checkout {tag}", shell=True)
    os.chdir(old_dir)
    
    return os.path.join(os.getcwd(), dir_name)

def download_gdrive(file_id):
    subprocess.run("wget https://raw.githubusercontent.com/GitHub30/gdrive.sh/master/gdrive.sh", shell=True)
    subprocess.run(f"curl gdrive.sh | bash -s {file_id}", shell=True)
    subprocess.run("rm gdrive.sh", shell=True)

def image_average(imlist, weights):
    assert len(imlist) == len(weights), "Input lists should have same size."
    weights = np.array(weights) / np.sum(weights)
    w, h = Image.open(imlist[0]).convert("RGB").size

    arr = np.zeros((h, w, 3), np.float)
    for im in imlist:
        imarr = np.array(Image.open(im), dtype=np.float)
        arr = arr + imarr / len(imlist)

    arr = np.array(np.round(arr), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")