# all functions and constants to be used. organize hierarchicaly
from modules import *

############# CONSTANTS AND PARAMETERS #################################
SMALL = np.finfo(float).eps
rho_air = 1.2 # [kg/m3]
ni = 14.88*10**-6 # [m2/s] kinematic viscosity air
kappa = 0.4 # von Karman constant for atmospheric flows

############# PLOTTING CUSTOMIZATIONS ##################################
plt.rcParams["figure.figsize"] = np.array(plt.rcParams["figure.figsize"])*1
fsize = min(plt.rcParams['figure.figsize'])
#plt.rcParams["image.cmap"] = 'Spectral'
plt.rcParams["image.cmap"] = 'gray' # for images
plt.rcParams["font.size"] = 18
plt.rcParams["axes.grid"] = False
plt.rcParams["figure.dpi"] = 200 # 100
markerlist = [ k for k in Line2D.markers.keys() ]
lineslist = [ k for k in Line2D.lineStyles.keys() ]
plt.rcParams['lines.markersize'] = 10
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.fancybox"] = False
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
ploting = True # disable if you don't want to see plots


############# TIER 1 ###################################################
def zero_to_small(A):
	"""
	Prevent division by zero by replacing near-zero values with SMALL.
	
	This is critical for calculations like atan(Im/Re) where Re might be zero.
	Values with absolute value smaller than machine epsilon are replaced with
	±SMALL to maintain sign while avoiding numerical issues.
	
	Parameters:
		A: numpy array
	
	Returns:
		A with near-zero values replaced by ±SMALL
	"""
	# Replace small positive values (0 to SMALL) with SMALL
	A[(A<SMALL) & (A>=0)] = SMALL
	# Replace small negative values (-SMALL to 0) with -SMALL
	A[(A>-SMALL) & (A<0)] = -SMALL
	return A

def poly_2d(x,y,K):
	"""
	Evaluate a 2D polynomial of degree 3 at points (x, y).
	
	Polynomial form:
		f(x,y) = K₀ + K₁x + K₂y + K₃x² + K₄xy + K₅y² + 
		         K₆x³ + K₇x²y + K₈xy² + K₉y³
	
	Used for coordinate transformations (e.g., pixel to physical coordinates).
	
	Parameters:
		x, y: coordinate arrays (can be scalars or arrays)
		K: coefficient array of length 10 [K₀, K₁, ..., K₉]
	
	Returns:
		Polynomial values at (x, y)
	"""
	return (K[0]+K[1]*x+K[2]*y + K[3]*x**2+K[4]*x*y+K[5]*y**2 + 
		K[6]*x**3+K[7]*x**2*y+K[8]*x*y**2+K[9]*y**3)

def fname_dir(fname):
	"""
	returns the directory in which fname is located
	"""
	return os.path.abspath(os.path.join(fname,os.pardir))

def df_add(df,index,column,value):
	"""
	Add a new value at column and index. If either column or index
	do not exist they are created.
	Works for pandas dataframes.
	"""
	try:
		df[column]
	except:
		df[column]=np.nan
	try:
		df.loc[index]
	except:
		df.loc[index]=np.nan
	df.loc[index,column]=value
	return df

def gaussian(x,mu,sigma,A):
	"""
	Evaluate a Gaussian (normal) distribution function.
	
	Mathematical form:
		G(x) = (A / (σ√(2π))) * exp(-(x-μ)²/(2σ²))
	
	Where:
		μ (mu): mean/center of the distribution
		σ (sigma): standard deviation (controls width)
		A: amplitude/scaling factor
	
	Parameters:
		x: input values (scalar or array)
		mu: mean of the distribution
		sigma: standard deviation
		A: amplitude
	
	Returns:
		Gaussian function values
	"""
	return A/(sigma*np.sqrt(2*np.pi))*np.exp(
		-(x-mu)**2/(2*sigma**2))

def read_external_data(fname,sep='	',coma=False,bn=False,header=0):
	"""
	read a data file located in fname structured in lines and columns 
	where each column is separated by sep. If the data uses comas 
	instead of dots put replace=True to sort compatibility. In case \n
	shows up at the end of each line but bn=True. Header is
	the number of lines to be skipped at the start of the file.
	returns a matrix
	"""
	f = open(fname,"r")
	Lines = f.readlines()[header:]
	N = len(Lines)
	nVal = len(Lines[N-1].split(sep)) # using last line as reference for number of cloumns
	A = np.zeros((N,nVal))
	for line in range(N):
		if coma:
			Lines[line] = Lines[line].replace(',' , '.')
		if bn:
			Lines[line] = Lines[line].replace('\n' , '')
		A[line] = np.array(Lines[line].split(sep))
	f.close()
	return A.transpose()

def detect_circles(img,resolution=1,delta=10,canny=30,akku=15,rmin=10,rmax=-1):
	"""
	circles: x,y vecotor
	resolution: when multiplied by image size
	delta: min space between circles in px
	canny: threshold for canny edge. The higher the stricter.
	akku: threshold for hough circles accumulator. The higher the stricter.
	rmin: minimum radius of circles
	"""
	circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,resolution,delta,
              param1=canny,param2=akku,minRadius=rmin,maxRadius=rmax)[0,:,:-1]
	return circles

def nan_interp(A):
	"""
	Interpolate NaN (Not a Number) values in a 2D matrix using nearest neighbors.
	
	Algorithm:
		1. Extend matrix edges by duplicating boundary values (to handle edge cases)
		2. For each NaN value, replace it with the mean of its 4 nearest neighbors
		   (up, down, left, right)
		3. If neighbors also contain NaNs, only average the valid neighbors
	
	This is a simple spatial interpolation method useful for filling gaps in image data.
	
	Parameters:
		A: 2D numpy array containing NaN values
	
	Returns:
		Array with NaN values replaced by interpolated values
	"""
	ni,nj = np.shape(A)
	# Extend edges by duplicating boundary values to handle edge cases
	# This creates a border so we can safely check neighbors
	A = np.concatenate((np.array([A[:,0]]).transpose(),A,np.array([A[:,-1]]).transpose()),axis=1)
	A = np.concatenate((np.array([A[0,:]]),A,np.array([A[-1,:]])),axis=0)
	
	# Find all NaN positions
	nanp = np.isnan(A)
	
	# For each NaN value, replace with mean of 4 nearest neighbors
	for i in range(1,ni+1):
		for j in range(1,nj+1):
			if nanp[i,j]:
				# Get 4 nearest neighbors: up, left, down, right
				b = np.array([A[i-1,j],A[i,j-1],A[i+1,j],A[i,j+1]])
				snan = np.sum(np.isnan(b))  # Count how many neighbors are also NaN
				sb = np.nansum(b)           # Sum of valid neighbors (ignores NaNs)
				# Replace NaN with mean of valid neighbors
				A[i,j] = sb/(len(b)-snan)
	
	# Extract the original-sized array (remove the extended borders)
	A = A[1:ni+1,1:nj+1]
	return A

def new_directory(name):
	if not os.path.isdir(name):
		os.mkdir(name)
	return name
	
def read_image(fname):
	"""
	Return the loaded image as is (with alpha channel)
	As a numpy array
	"""
	img = cv2.imread(fname,cv2.IMREAD_GRAYSCALE)
	return img

def dir_ftype(directory,extension):
	"""
	find all files with given extension in directory
	order them alphabetically in a list and return them
	"""
	extension = extension.replace(".","") # make sure theres no "."
	fnames = directory+os.sep+"*"+"."+extension
	fnames = glob.glob(fnames)
	fnames = np.sort(fnames) # order files from 0 to last
	return fnames

def dir_fname(directory,nametype):
	"""
	find all files with given extension and string in directory
	order them alphabetically in a list and return them
	"""
	fnames = directory+os.sep+nametype
	fnames = glob.glob(fnames)
	fnames = np.sort(fnames) # order files from 0 to last
	return fnames

def crop_image(img,roi):
	"""
	roi: region of interest, numpy slice, 4 elements, 
	[y_min:y_max,x_min:x_max]
	"""
	img = img[roi]
	return img
	
def canny_edge(img,threshold=15,hysteresis=5):
	img = cv2.Canny(img, threshold-hysteresis,threshold+hysteresis)
	return img

def blur_image(img,Ks,strength=1):
	"""
	Apply Gaussian blur to an image using OpenCV.
	
	Gaussian blur reduces noise and smooths the image by convolving with a
	Gaussian kernel. The blur is controlled by:
		- Kernel size (Ks): larger = more blur, must be odd
		- Sigma (strength): standard deviation of Gaussian, controls blur intensity
	
	Mathematical operation: I_blurred = I * G(σ), where G is 2D Gaussian kernel
	
	Parameters:
		img: input grayscale image (2D numpy array)
		Ks: kernel size (must be odd, e.g., 3, 5, 7, 9...)
		strength: sigma parameter for Gaussian (standard deviation)
	
	Returns:
		Blurred image
	"""
	Ks = int(Ks)
	if Ks%2 != 1:
		print("blur_image: Ks must be odd! Continuing with Ks = Ks-1")
		Ks = Ks-1
	# Apply 2D Gaussian blur: (kernel_width, kernel_height), sigma
	img = cv2.GaussianBlur(img,(Ks,Ks),strength)
	return img

def blurd_image(img,order=5,direction='horizontal',strength=0.25,speed='slow'):
	"""
	Apply directional blur to an image (blur in one direction only).
	
	Two methods available:
		'slow': Butterworth low-pass filter - smooth frequency-domain filtering
		'fast': Moving average convolution - simple spatial averaging
	
	This is useful for removing noise while preserving edges perpendicular to
	the blur direction (e.g., blur horizontally to preserve vertical edges).
	
	Parameters:
		img: input image (2D numpy array)
		order: filter order for Butterworth (higher = sharper cutoff)
		direction: 'horizontal' or 'vertical' blur direction
		strength: cutoff frequency (Butterworth) or kernel size (moving average)
		speed: 'slow' (Butterworth) or 'fast' (moving average)
	
	Returns:
		Directionally blurred image
	"""
	ny,nx = np.shape(img)
	if speed == 'slow':
		# Butterworth low-pass filter: smooth frequency-domain filtering
		# Creates filter coefficients for order-th order Butterworth filter
		b, a = scig.butter(order, strength)
		if direction == 'horizontal':
			# Apply filter to each row (blur horizontally)
			for i in range(ny):
				# filtfilt applies filter forward and backward (zero-phase)
				img[i,:] = scig.filtfilt(b, a, img[i,:])
		elif direction == 'vertical':
			# Apply filter to each column (blur vertically)
			for i in range(nx):
				img[:,i] = scig.filtfilt(b, a, img[:,i])
		return img
	elif speed=='fast':
		# Moving average: simple convolution with uniform kernel
		# Create normalized averaging kernel of size 'strength'
		K = np.ones((int(strength),1))
		K = K/np.sum(K)  # Normalize so sum = 1 (preserves image brightness)
		if direction=='vertical':
			# Convolve with vertical kernel (blur vertically)
			img=scig.convolve2d(img,K,mode='same')
		elif direction=='horizontal':
			# Convolve with horizontal kernel (blur horizontally)
			img=scig.convolve2d(img,K.transpose(),mode='same')
		return img
			
			

def zernike_Vnm(rho,theta,n,m):
	"""
	Calculate Zernike polynomial V_nm(ρ,θ) for given radial and angular orders.
	
	Mathematical form:
		V_nm(ρ,θ) = R_nm(ρ) * exp(jmθ)
	
	Where R_nm(ρ) is the radial polynomial:
		R_nm(ρ) = Σ[(-1)^s * (n-s)! * ρ^(n-2s)] / 
		          [s! * ((n+|m|)/2 - s)! * ((n-|m|)/2 - s)!]
	
	This is the general Zernike polynomial formula. For edge detection, we
	typically only need Z₁₁ and Z₂₀, which are computed directly in ghosal_edge_v2.
	
	Parameters:
		rho: radial coordinate (0 to 1, normalized to unit circle)
		theta: angular coordinate (0 to 2π)
		n: radial order (non-negative integer)
		m: azimuthal frequency (integer, |m| ≤ n, n-|m| even)
	
	Returns:
		Vnm: Complex value of Zernike polynomial
	"""
	Rnm = 0
	fact = lambda x: np.math.factorial(x)
	am = abs(m)
	# Sum over s from 0 to (n-|m|)/2 to compute radial polynomial R_nm
	for s in range(0,(n-am)/2):
		Rnm+= (-1)**s*fact(n-s)*rho**(n-2*s)/(
			fact(s)*fact((n+am)/2-s)*fact((n-am)/2-s))
	# Multiply radial polynomial by angular component
	Vnm = Rnm*np.exp(1j*m*theta)
	return Vnm  # Note: function was missing return statement


def ddx(a):
	"""
	Calculate the first derivative of a 1D array using central difference method.
	
	Mathematical method: Central difference approximation
		f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
	
	For unit spacing (h=1), this becomes:
		f'(i) ≈ [f(i+1) - f(i-1)] / 2
	
	The kernel [-0.5, 0, 0.5] implements this: output[i] = -0.5*a[i-1] + 0.5*a[i+1]
	
	Edge handling: Extends array by mirroring boundary values to avoid edge artifacts.
	
	Parameters:
		a: 1D numpy array
	
	Returns:
		First derivative array (same length as input after edge extension removal)
	"""
	# Extend array by mirroring edges to avoid boundary effects
	# This creates symmetric padding: [a[1], a[0], a, a[-1], a[-2]]
	thick = 2
	a = np.concatenate((a[(thick-1)::-1],a,a[:-(thick+1):-1]))
	
	# Central difference kernel: [-0.5, 0, 0.5]
	# This computes: f'(i) = (f(i+1) - f(i-1)) / 2
	K = np.array([-0.5,0,0.5])
	mode = "valid"  # Return only valid (non-padded) results
	da = scig.convolve(a,K,mode=mode)
	return da

############# ZERNIKE MOMENTS EDGE DETECTION ##########################
# Implementation of sub-pixel edge detection using Zernike moments
# Based on Ghosal algorithm for planetary limb detection
# Key mathematical components:
#   - Zernike polynomial kernels (Z₁₁, Z₂₀)
#   - Zernike moments calculation via convolution
#   - Edge angle extraction: φ = arg(A₁₁)
#   - Edge distance: l = A₂₀/A'₁₁
#   - Sub-pixel position conversion
# KEY FUNCTION
def ghosal_edge(img,Ks,thr=1,thrmax=0.995,lmin = 0.5,phimin=1.4,thresholding=True, debug=False):
	"""
	implementation of the subpixel edge detection method of [2]. The
	pixels are projected into a new orthogonal domain where a parameter
	k defines the intesity of the edge. By filtering out edges of small
	k the relevant ones remain. The extracted parameters are enough to 
	define a straight edge.
	img: the image to be treated.
	Ks: kernel size
	thr: threshold paramter limiting the minimum of k
	thrmax: threshold limiting the maximum of k. This is usefull to
		remove reflections which have excessively sharp edges
	lmin: is the minimum distance between the pixel center and the
		detected edge. This avoids that big kernels do errors, when 
		close to multiple edges. It also avoids values of l which are
		nonesense, since they only make sense for real edges.
	phimin: allows the user to define a minimum angle in redians 
		measured between the y-axis and the edge.
	"""
	totaltime = time.time()
	kerneltime = time.time()
	# Ks must be odd
	if Ks%2 != 1:
		print("Ks must be odd! Continuing with Ks = Ks-1")
		Ks = Ks-1
	# define the rectangular kernels
	#Vc00 = np.zeros((Ks,Ks),dtype=complex)
	Vc11 = np.zeros((Ks,Ks),dtype=complex)
	Vc20 = np.zeros((Ks,Ks),dtype=complex)
	ofs = 1 *(1-1/Ks) # offset for centering kernel around 0,0
	for i in range(Ks):
		for j in range(Ks):
			Kx = 2*j/Ks-ofs # limits of integration between -1 and 1
			Ky = 2*i/Ks-ofs
			if Kx**2+Ky**2 <= 1: # only a circle
				#Vc00[i,j] = 1 # the conjugate of V00
				Vc11[i,j] = Kx-Ky*1j # ...
				Vc20[i,j] = 2*Kx**2+2*Ky**2-1
	kerneltime = time.time() - kerneltime
	
	# Kernel Plots
	#	VCplot = Vc00
	#	plt.pcolormesh(np.real(VCplot))
	#	plt.title("real w K Vc00")
	#	plt.colorbar()
	#	plt.show()
	#	plt.pcolormesh(np.imag(VCplot))
	#	plt.title("imag w K Vc00")
	#	plt.colorbar()
	#	plt.show()
	#	VCplot = Vc11
	#	plt.pcolormesh(np.real(VCplot))
	#	plt.title("real w K Vc11")
	#	plt.colorbar()
	#	plt.show()
	#	plt.pcolormesh(np.imag(VCplot))
	#	plt.title("imag w K Vc11")
	#	plt.colorbar()
	#	plt.show()
	#	VCplot = Vc20
	#	plt.pcolormesh(np.real(VCplot))
	#	plt.title("real w K Vc20")
	#	plt.colorbar()
	#	plt.show()
	#	plt.pcolormesh(np.imag(VCplot))
	#	plt.title("imag w K Vc20")
	#	plt.colorbar()
	#	plt.show()
	
	# do the convolution with the images to get the zernike moments
	Anorm = lambda n : (n+1)/np.pi	# a normalization value
	convolvetime = time.time()
	#A00 = scig.convolve2d(img,Vc00,mode='same')
	#	A11 = Anorm(1)*scig.convolve2d(img,Vc11,mode='same')
	#	A20 = Anorm(2)*scig.convolve2d(img,Vc20,mode='same')
	A11 = Anorm(1)*scig.oaconvolve(img,Vc11,mode='same')
	A20 = Anorm(2)*scig.oaconvolve(img,Vc20,mode='same')
	convolvetime = time.time() - convolvetime
	# Plot Zernike moments
	#	VCplot = A00
	#	plt.pcolormesh(np.real(VCplot))
	#	plt.title("real A00")
	#	plt.colorbar()
	#	plt.show()
	#	plt.pcolormesh(np.imag(VCplot))
	#	plt.title("imag A00")
	#	plt.colorbar()
	#	plt.show()
	#	VCplot = A11
	#	plt.pcolormesh(np.real(VCplot))
	#	plt.title("real A11")
	#	plt.colorbar()
	#	plt.show()
	#	plt.pcolormesh(np.imag(VCplot))
	#	plt.title("imag A11")
	#	plt.colorbar()
	#	plt.show()
	#	VCplot = A20
	#	plt.pcolormesh(np.real(VCplot))
	#	plt.title("real A20")
	#	plt.colorbar()
	#	plt.show()
	#	plt.pcolormesh(np.imag(VCplot))
	#	plt.title("imag A20")
	#	plt.colorbar()
	#	plt.show()
	
	paramstime = time.time()
	# calculate the edge paramters
	#	tanphi = np.imag(A11)/np.real(A11)
	#	phi = np.arctan(tanphi)
	#	cosphi = np.cos(phi)
	#	sinphi = cosphi*tanphi
	#	Al11 = np.real(A11)*cosphi+np.imag(A11)*sinphi
	
	phi = np.arctan(np.imag(A11)/np.real(A11))
	Al11 = np.real(A11)*np.cos(phi)+np.imag(A11)*np.sin(phi)
	
	#	Al11 = A11*np.exp(-phi*1j)
	l = A20/Al11 # A20 has no imaginary component so A20 = A'20

	k = 3*Al11/(2*(1-l**2)**(3/2))
	paramstime = time.time() - paramstime
	
	# Plot edge paramters
	#	VCplot = phi
	#	plt.pcolormesh(np.real(VCplot))
	#	plt.title("real phi")
	#	plt.colorbar()
	#	plt.show()
	#	plt.pcolormesh(np.imag(VCplot))
	#	plt.title("imag phi")
	#	plt.colorbar()
	#	plt.show()
	#	VCplot = Al11
	#	plt.pcolormesh(np.real(VCplot))
	#	plt.title("real A\'11")
	#	plt.colorbar()
	#	plt.show()
	#	plt.pcolormesh(np.imag(VCplot))
	#	plt.title("imag A\'11")
	#	plt.colorbar()
	#	plt.show()
	#	VCplot = l
	#	plt.pcolormesh(np.real(VCplot))#,vmin=-5,vmax=5
	#	plt.title("real l")
	#	plt.colorbar()
	#	plt.show()
	#	plt.pcolormesh(np.imag(VCplot)) # ,vmin=-5,vmax=5
	#	plt.title("imag l")
	#	plt.colorbar()
	#	plt.show()
	#	VCplot = k
	#	plt.pcolormesh(np.real(VCplot))
	#	plt.title("real k")
	#	plt.colorbar()
	#	plt.show()
	#	plt.pcolormesh(np.imag(VCplot))
	#	plt.title("imag k")
	#	plt.colorbar()
	#	plt.show()
	
	
	treattime = time.time()
	# ====================================================================
	# FILTER EDGE DETECTIONS BY QUALITY CRITERIA
	# ====================================================================
	# Apply adaptive thresholds based on edge strength (k) distribution
	# This removes false positives and keeps only reliable edge points
	
	if thresholding==True:
		# Calculate percentile-based thresholds from k distribution
		# This adapts to the image's edge strength characteristics
		
		if (thrmax<0)&(thr>0):
			# Case 1: thr is percentile, thrmax is negative (ignore upper bound)
			# Find k value at thr-th percentile
			knorm = np.sort(k.flatten())[[int(thr*np.size(k)),int(thrmax*np.size(k))]]
			# Keep edges with: |l| < lmin, |phi| > phimin, k > threshold
			idx = (abs(l)<lmin)&(abs(phi)>phimin)&(abs(k)>knorm[0])
		elif thrmax>0:
			# Case 2: Both thr and thrmax are percentiles (two-sided threshold)
			# Find k values at thr-th and thrmax-th percentiles
			knorm = np.sort(k.flatten())[[int(thr*np.size(k)),int(thrmax*np.size(k))]]
			# Keep edges with: |l| < lmin, |phi| > phimin, kmin < k < kmax
			idx = (abs(l)<lmin)&(abs(phi)>phimin)&(abs(k)>knorm[0])&(abs(k)<knorm[1])
		elif thr<0:
			# Case 3: thr is negative index (absolute threshold value)
			# First filter by l and phi, then apply k threshold
			idx = (abs(l)<lmin)&(abs(phi)>phimin)
			# Find k threshold from filtered points
			knorm = np.sort(k[idx].flatten())[int(thr)]
			# Apply k threshold
			idx = idx&(abs(k)>abs(knorm))
		ne = np.sum(idx)  # Count of valid edge points
	elif thresholding==False:
		raise ValueError("this option is not still uncer development")
		# no thresholding
		idx = np.ones(np.shape(l),dtype=bool)
		ne =np.sum(idx)
	else:
		raise ValueError("thresholding should be boolean")
	
	# ====================================================================
	# CONVERT TO SUB-PIXEL EDGE POSITIONS
	# ====================================================================
	# Transform from unit circle coordinates (l, φ) back to pixel coordinates
	# Formula: [u_i; v_i] = [ũ_i; ṽ_i] + (N*l/2) * [cos(φ); sin(φ)]
	
	# Initialize output arrays for edge positions and original pixel centers
	edg = np.zeros((ne,2))  # Sub-pixel edge positions
	org = np.zeros((ne,2))  # Original pixel centers
	nx,ny = np.shape(img)
	e = 0  # Counter for valid edge points
	
	# Loop through all pixels and extract valid edge detections
	for i in range(nx):
		for j in range(ny):
			if idx[i,j]:  # If this pixel has a valid edge detection
				# Convert from unit circle coordinates to pixel coordinates
				# l*Ks/2 scales distance from unit circle to pixels
				# [sin(φ), -cos(φ)] is the direction vector (note: -cos for y-axis convention)
				edg[e] = np.array([i,j]) + l[i,j]*Ks/2*np.array(
					[np.sin(phi[i,j]),-np.cos(phi[i,j])])
				org[e] = np.array([i,j])  # Store original pixel center
				e += 1
	treattime = time.time() - treattime
	totaltime = time.time() - totaltime
	print("total %0.5f	convolution %0.5f	thresholding %0.5f	paramters %0.5f	kernel %0.5f"%(totaltime,convolvetime,treattime,paramstime,kerneltime))
	
	if debug==True:
		return edg, org, k, l, phi
	else:
		return edg, org
		
def ghosal_edge_v2(img,Ks,kmin=0,kmax=1000,lmax=0.5,phimin=1,thresholding=True,debug=False,mirror=False):
	"""
	implementation of the subpixel edge detection method of [2]. The
	pixels are projected into a new orthogonal domain where a parameter
	k defines the intesity of the edge. By filtering out edges of small
	k the relevant ones remain. The extracted parameters are enough to 
	define a straight edge.
	img: the image to be treated.
	Ks: kernel size
	thr: threshold paramter limiting the minimum of k
	kmax/min: threshold limiting k. This is usefull to define the edge
		intensity range that we are looking for. Neither too strong nor
		too weak
	lmax: is the maximum distance between the pixel center and the
		detected edge. This avoids that big kernels do errors, when 
		close to multiple edges. It also avoids values of l which are
		nonesense, since they only make sense for real edges.
	phimin: allows the user to define a minimum angle in radians 
		measured between the y-axis and the edge.
	thresholding: in case no thresholding is desired, might make sense
		for debugging or for post-processing raw data of edges
	debug: will output k, l and phi matrices
	mirror: mirror the limits of the image during convolution 
		so that no aliasing happens during the convolution. 
		Convolution time is doubled for some cases if this is activated. 
	"""
	# gather image properties before its altered
	ni,nj = np.shape(img)
	# Ks must be odd
	if Ks%2 != 1:
		print("Ks must be odd! Continuing with Ks = Ks-1")
		Ks = Ks-1
	# ====================================================================
	# STEP 1: CONSTRUCT ZERNIKE POLYNOMIAL KERNELS
	# ====================================================================
	# Create kernels for Zernike polynomials Z₁₁ and Z₂₀
	# These kernels represent the Zernike basis functions on a unit circle
	# Reference: Zernike moments for sub-pixel edge detection
	
	Vc11 = np.zeros((Ks,Ks),dtype=complex)  # Z₁₁ kernel (complex)
	Vc20 = np.zeros((Ks,Ks),dtype=complex)  # Z₂₀ kernel (real, stored as complex)
	ofs = 1 *(1-1/Ks)  # offset for centering kernel around (0,0)
	
	for i in range(Ks):
		for j in range(Ks):
			# Normalize pixel coordinates to unit disk [-1, 1]
			Kx = 2*j/Ks - ofs  # x-coordinate in unit circle
			Ky = 2*i/Ks - ofs   # y-coordinate in unit circle
			
			# Only compute within unit circle (Zernike polynomials are defined on unit disk)
			if Kx**2 + Ky**2 <= 1:
				# Z₁₁ polynomial: T₁₁(r,θ) = r * exp(jθ) = (x + jy) = Kx - j*Ky
				# This gives edge orientation information
				Vc11[i,j] = Kx - Ky*1j
				
				# Z₂₀ polynomial: T₂₀(r,θ) = 2r² - 1 = 2(x²+y²) - 1
				# This gives radial distance information
				Vc20[i,j] = 2*Kx**2 + 2*Ky**2 - 1
	# Mirror image edges to avoid convolution artifacts at boundaries
	# When convolving near edges, we need padding. Mirroring preserves
	# edge continuity better than zero-padding or constant padding.
	if mirror:
		thick = int((Ks-1)/2)  # Padding thickness = half kernel size
		# Mirror horizontally: [flipped_left, original, flipped_right]
		img = np.concatenate((img[:,(thick-1)::-1],img,img[:,:-(thick+1):-1]),1)
		# Mirror vertically: [flipped_top, original, flipped_bottom]
		img = np.concatenate((img[(thick-1)::-1,:],img,img[:-(thick+1):-1,:]),0)
		mode = "valid"  # Return only valid (non-padded) region
	else:
		mode = "same"   # Return same size as input (uses zero-padding)
	
	# ====================================================================
	# STEP 2: COMPUTE ZERNIKE MOMENTS
	# ====================================================================
	# Calculate Zernike moments A₁₁ and A₂₀ by convolving image with kernels
	# Formula: Z_nm = (n+1)/π * A_nm, where A_nm = ∫∫ f(u,v) * T_nm(u,v) du dv
	# In discrete form: A_nm ≈ Σ Σ I(u,v) * M_nm(u,v) (convolution)
	
	Anorm = lambda n : (n+1)/np.pi  # Normalization factor: (n+1)/π
	
	# Compute Zernike moments via convolution
	# A₁₁: Complex moment encoding edge orientation and strength
	A11 = Anorm(1) * scig.oaconvolve(img, Vc11, mode=mode)  # Z₁₁ moment
	
	# A₂₀: Real moment encoding radial distance to edge
	A20 = Anorm(2) * scig.oaconvolve(img, Vc20, mode=mode)  # Z₂₀ moment

	# ====================================================================
	# STEP 3: EXTRACT EDGE PARAMETERS FROM ZERNIKE MOMENTS
	# ====================================================================
	
	# Calculate edge angle φ (psi in notes)
	# φ = arg(A₁₁) = atan2(Im(A₁₁), Re(A₁₁))
	# This gives the orientation of the edge at each pixel
	phi = np.arctan(np.imag(A11) / zero_to_small(np.real(A11)))
	
	# Rotate A₁₁ to align with edge direction: A'₁₁ = Re(A₁₁)*cos(φ) + Im(A₁₁)*sin(φ)
	# This is equivalent to: A'₁₁ = A₁₁ * exp(-jφ) (rotation to edge-aligned coordinates)
	Al11 = np.real(A11)*np.cos(phi) + np.imag(A11)*np.sin(phi)
	
	# Calculate edge distance l from pixel center
	# l = A₂₀ / A'₁₁ (distance from center to edge in unit circle coordinates)
	# Note: A₂₀ is purely real, so A₂₀ = A'₂₀
	l = np.real(A20) / Al11
	
	# Clamp l to valid range [-1, 1] (must be within unit circle)
	l = np.minimum(l, 1-SMALL)
	l = np.maximum(l, -1+SMALL)
	
	# Calculate edge strength parameter k
	# k = 3*A'₁₁ / (2*(1-l²)^(3/2))
	# This measures the intensity contrast across the edge
	k = abs(3*Al11 / (2*(1-l**2)**(3/2))) 
	
	# ====================================================================
	# STEP 4: FILTER EDGE DETECTIONS BY QUALITY CRITERIA
	# ====================================================================
	# Apply thresholds to filter out low-quality or invalid edge detections
	# This removes false positives and keeps only reliable edge points
	
	if thresholding==True:
		# Create boolean masks for each quality criterion:
		# phi_c: Edge angle must be significant (avoid near-horizontal edges)
		phi_c = abs(phi)>phimin
		
		# l_c: Edge must be within kernel radius (|l| < lmax)
		# Large |l| means edge is far from pixel center, less reliable
		l_c = abs(l)<lmax
		
		# k_c: Edge strength must be in valid range (kmin < k < kmax)
		# k too small = weak edge (noise), k too large = oversaturated/reflection
		k_c = (k<kmax) & (k>kmin)
		
		# Combine all conditions: edge must satisfy ALL criteria
		valid = phi_c & (k_c & l_c)
	elif thresholding==False:
		# No filtering: accept all detected edges (useful for debugging)
		valid = np.ones_like(k)
	# Create coordinate grids for all pixels in the image
	# meshgrid creates 2D arrays where i[i,j] = row index, j[i,j] = column index
	i,j = np.meshgrid(np.arange(nj),np.arange(ni))
	
	# Extract only the coordinates and parameters for valid edge detections
	# This converts from 2D arrays to 1D lists of valid edge points
	i = i[valid]  # Row indices of valid edge pixels
	j = j[valid]  # Column indices of valid edge pixels
	#	k = k[valid] # Edge strength (not needed for final output)
	l = l[valid]    # Edge distances for valid points
	phi = phi[valid]  # Edge angles for valid points
	
	# ====================================================================
	# STEP 4: CONVERT TO SUB-PIXEL EDGE POSITIONS
	# ====================================================================
	# Transform from unit circle coordinates (l, φ) back to pixel coordinates
	# Formula from notes: [u_i; v_i] = [ũ_i; ṽ_i] + (N*l/2) * [cos(φ); sin(φ)]
	# where N = Ks (kernel size), l is distance, φ is angle
	
	# Convert polar coordinates (l, φ) to pixel offset
	# Scale l from unit circle [-1,1] to pixel units: multiply by Ks/2
	i_s = i + l*Ks/2 * np.cos(phi)  # Sub-pixel row position
	j_s = j + l*Ks/2 * np.sin(phi)  # Sub-pixel column position
	
	# Format output: edge positions and original pixel centers
	# Note: Coordinate convention: (j, i) = (x, y) = (column, row)
	edg = np.squeeze((j_s, i_s)).transpose()  # Final sub-pixel edge positions
	org = np.squeeze((j, i)).transpose()      # Original pixel centers
	if debug==True:
		return edg, org, k, l, phi
	else:
		return edg, org

def line_fit(x,y):
	"""
	Fit a straight line to a set of (x,y) points using least squares.
	
	Mathematical method: Linear least squares regression
		Find m, b that minimize: Σ(y_i - (m*x_i + b))²
	
	The solution uses scipy's curve_fit to find optimal slope (m) and
	intercept (b) for the line y = m*x + b.
	
	Parameters:
		x: x-coordinates (1D array)
		y: y-coordinates (1D array)
	
	Returns:
		pts: Two points defining the fitted line [[x_min, y_min], [x_max, y_max]]
		sig: Standard deviation of residuals (measure of fit quality)
	"""
	# Ensure inputs are 1D arrays
	x = np.squeeze(x)
	y = np.squeeze(y)
	
	# Combine into N×2 array: [[x₁,y₁], [x₂,y₂], ...]
	xy = np.concatenate((x[:,np.newaxis],y[:,np.newaxis]),1)
	
	# Sort by x-coordinate for consistent output
	xy = xy[xy[:,0].argsort()]
	
	# Define linear function: y = m*x + b
	f = lambda x,m,b : m*x+b
	
	# Fit line using least squares (minimizes sum of squared residuals)
	pars,_ = opt.curve_fit(f,xy[:,0],xy[:,1])
	m = pars[0]  # Slope
	b = pars[1]  # Intercept
	
	# Create two points spanning the x-range of the data
	pts = np.zeros((2,2))
	pts[0,0] = xy[0,0]      # x_min
	pts[1,0] = xy[-1,0]     # x_max
	pts[:,1] = pts[:,0]*m+b  # y values from fitted line
	
	# Calculate residual standard deviation (measure of scatter around line)
	sig = np.std((xy[:,1]-f(xy[:,0],m,b)))
	return pts, sig


def remove_dbledge(img):
	"""
	The image has to be binary. Get only the points closest to the 
	bottom of the image.
	"""
	(ny,nx) = np.shape(img)
	for i in range(nx):
		idx = np.array(np.nonzero(img[:,i]))
		if np.size(idx) == 0:
			continue
		idx = idx[0][-1]
		img[idx-1::-1,i] = 0
	return img

def solve_L(centers_i,centers_r):
	"""
	Solve for camera calibration rotation and translation parameters.
	
	Mathematical method: Linear least squares solution to calibration problem
		L * [r'₁, r'₂, T'x, r'₄, r'₅]ᵀ = b
	
	This is an intermediate step in camera calibration that solves for
	rotation matrix elements and translation components from known
	correspondences between image coordinates (centers_i) and reference
	coordinates (centers_r).
	
	The system is derived from the camera projection model:
		x_image = f(rotation_matrix * x_reference + translation)
	
	Parameters:
		centers_i: N×2 array of image coordinates (detected points)
		centers_r: N×2 array of reference coordinates (known positions)
	
	Returns:
		rl: Solution vector [r'₁, r'₂, T'x, r'₄, r'₅] containing rotation
		    and translation parameters
	"""
	# Build the design matrix L for the linear system
	# Each row corresponds to one correspondence point
	
	# Ly: contribution from y-coordinates
	# Multiplies y_image with reference coordinates
	Ly = centers_i[:,1][np.newaxis,:].transpose() * centers_r
	
	# Lx: contribution from x-coordinates (with sign flip)
	# Multiplies -x_image with reference coordinates
	Lx = - centers_i[:,0][np.newaxis,:].transpose() * centers_r
	
	# Combine: L = [Ly, y_image_column, Lx]
	# This creates a matrix where each row encodes the relationship
	# between image and reference coordinates for one point
	L = np.concatenate((Ly,centers_i[:,1][np.newaxis,:].transpose(),Lx),
		axis=1)
	
	# Right-hand side: x-coordinates of image points
	b = centers_i[:,0]
	
	print("solving for the rotation and translation coefficients...")
	# Solve linear least squares: L * rl = b
	rl,resids,rank,svals = np.linalg.lstsq(L,b)
	print("residue:%0.4f	rank:%0.4f"%(np.sum(resids),rank))
	return rl

############# TIER 2 ###################################################
def select_points(fname,n=-1,prompt=""):
	"""
	Open an image and allow for the selection of points to be diplayed
	as an array later
	"""
	#def onclose(event):
	#	raise ValueError("Figure Closed by user. No input. Exiting.")
	if type(fname) is type(plt.figure()):
		fig = fname
		ax1 = fig.axes[0]
	else:
		fig,ax1 = plt.subplots(1)
	
	if type(fname) is str:
		img = read_image(fname)
	elif type(fname) is np.ndarray:
		img = fname
	#if n == 1:
	#	mouse = {"mouse_add":2, "mouse_pop":1,"mouse_stop":3}
	#	plt.xlabel("middle click: add point",fontsize="xx-small")
	#else:
	mouse = {"mouse_add":2, "mouse_pop":1,"mouse_stop":3}
	try:
		ax1.imshow(img)
	except:
		1 == 2
	fig.suptitle(prompt)
	ax1.set_xlabel("L click: remove previous - middle click: add point - R click: exit",fontsize="xx-small")
	#fig = plt.gcf() # get current figure
	#fig.canvas.mpl_connect('close_event', onclose) # raise value error if figure is closed
	pts = fig.ginput(n=n,timeout=0,**mouse)
	plt.close()
	return pts

def gradient_edge(img):
	def error(y,mu,sigma,A):
		return y-gaussian(np.arange(0,len(y)),mu,sigma,A)
	#	optimum,residu = opt.leastsq(\
	#		error, guess, args=(x,y))
	nx,ny = np.shape(img)
	
	for i in range(nx):
		I = img[:,i]
		dI = ddx(I)
		maxx = np.where(dI == max(dI))
		minx = np.where(dI == min(dI))
		plt.plot(I)
		plt.plot(dI)
		plt.show()
		#if i == 0:

def polyfit_2d(Xu,X):
	"""
	Fit a 2D polynomial transformation from pixel coordinates to physical coordinates.
	
	Mathematical method: Polynomial least squares
		Given: pixel coordinates (xu, yu) and physical coordinates X
		Find: coefficients K such that X ≈ poly_2d(xu, yu, K)
	
	The polynomial has 10 terms (degree 3):
		X = K₀ + K₁xu + K₂yu + K₃xu² + K₄xu*yu + K₅yu² + 
		    K₆xu³ + K₇xu²*yu + K₈xu*yu² + K₉yu³
	
	This solves the linear system: M * K = X, where M is the design matrix
	containing polynomial basis functions evaluated at each point.
	
	Parameters:
		Xu: N×2 array of pixel coordinates [[xu₁,yu₁], [xu₂,yu₂], ...]
		X: N×1 array of physical coordinates [X₁, X₂, ..., Xₙ]
	
	Returns:
		K: 10-element coefficient array [K₀, K₁, ..., K₉]
	"""
	xu = Xu[:,0]  # Extract x-coordinates
	yu = Xu[:,1]  # Extract y-coordinates
	X = np.squeeze(X)  # Ensure X is 1D array
	
	# Build design matrix M: each row contains polynomial basis functions
	# evaluated at one point (xu, yu)
	# Columns: [1, xu, yu, xu², xu*yu, yu², xu³, xu²*yu, xu*yu², yu³]
	M = np.squeeze((np.ones(xu.size),xu,yu,xu**2,xu*yu,yu**2,
		xu**3,xu**2*yu,xu*yu**2,yu**3))
	M = M.transpose()  # Shape: N×10 (N points, 10 coefficients)
	
	print("solving for the polynomial fitting coefficients...")
	# Solve least squares: minimize ||M*K - X||²
	# This finds K that best fits the transformation
	K,resid,rnk,svs = np.linalg.lstsq(M,X,rcond=-1)
	print("residue:%0.8f	rank:%0.8f"%(np.sum(resid),rnk))
	return K
	
	
def beads(img,yps=None,thr=6,skip=5,nbead=0):
	ny,nx = np.shape(img)
	
	def search_peaks(yps):
		dI = ddx(img[yps,:])
		peaks = np.zeros(len(dI))
		i = 0
		while i < len(dI):
				if dI[i]>thr:
					peaks[i] = 1
					i+=skip
				i+=1
		return peaks, dI # a vector with 1 at positions where a peak was detected		
	if yps == None: # scan the image vertically until a good result is found
		yps=0
		peaks = 0
		while np.sum(peaks)!=nbead:
			yps+=1
			peaks, dI = search_peaks(yps)
	else:
		peaks, dI = search_peaks(yps)
	#	plt.plot(peaks)
	#	plt.plot(dI)
	#	plt.show()
	return peaks, dI, yps


############# TIER 3 ###################################################

def tune_ghosal(fimg,K_s=5,k_m=None,N_k=5000,l_max=0.5,phi_min=1.45,outlier_sigma=2,blur=10):
	"""
	Interactive tool to tune Ghosal edge detection parameters.
	
	This function helps find optimal parameters (k_min, k_max, etc.) by:
		1. Running edge detection on the image
		2. Allowing user to select a characteristic edge point
		3. Automatically determining k thresholds around that point
		4. Removing outliers using statistical filtering
		5. Visualizing and optionally saving the parameters
	
	Parameters:
		fimg: path to image file
		K_s: kernel size for Zernike moments
		k_m: target edge strength (if None, user selects interactively)
		N_k: number of edge points to include around k_m (defines k range)
		l_max: maximum edge distance from pixel center
		phi_min: minimum edge angle
		outlier_sigma: standard deviations for outlier removal
		blur: blur strength for preprocessing
	
	Returns:
		None (saves parameters to file if user confirms)
	"""
	# Load image
	img = read_image(fimg)
	camera = fimg[-10]  # Extract camera identifier from filename
	
	# Ensure N_k is even (for symmetric range around k_m)
	if N_k%2!=0:
		N_k+=1
		print("N_k=%0.5f"%N_k)
	
	# Preprocess: apply directional blur to reduce noise
	img = blurd_image(img,order=1,strength=blur,speed='fast')
	
	# Run edge detection to get full parameter maps (k, l, phi)
	_,_,k,l,phi = ghosal_edge_v2(img,Ks=K_s,debug=True)
	
	# Find target edge strength k_m
	if k_m == None:
		# User interactively selects a characteristic edge point
		ij_m = select_points(img,n=1,prompt="choose a characteristic edge location")
		ij_m = np.array(ij_m[0],dtype=int)
		# Extract k value at selected point
		k_m = k[ij_m[1],ij_m[0]]
		print("k_m=",k_m)
	
	# Determine k_min and k_max by finding k_m in sorted distribution
	# and taking N_k points around it
	k_sort = np.sort(k.flatten())
	k_sort[np.isnan(k_sort)]=0  # Replace NaNs with 0 for argmin
	
	# Find index of k_m in sorted array (closest value)
	i_km = (np.abs(k_sort - k_m)).argmin()
	
	# Define symmetric range: N_k/2 points on each side of k_m
	i_kmin = int(i_km-N_k/2)
	i_kmax = int(i_km+N_k/2)
	k_min = k_sort[i_kmin]  # Lower threshold
	k_max = k_sort[i_kmax]  # Upper threshold
	
	# Re-run edge detection with tuned parameters
	edg,org = ghosal_edge_v2(img,K_s,kmax=k_max,kmin=k_min,lmax=l_max,phimin=phi_min)
	
	# Remove outliers using statistical filtering
	# Fit a line to the edge points and remove points far from the line
	pts,sig = line_fit(edg[:,1],edg[:,0])  # Fit line: x vs y
	ptsy = np.mean(pts[:,1])  # Mean y-coordinate of fitted line
	
	# Keep only points within outlier_sigma standard deviations
	# This removes spurious detections that don't follow the edge
	accepted = ((edg[:,0]<(ptsy+outlier_sigma*sig)) & (edg[:,0]>(ptsy-outlier_sigma*sig)))
	edga = edg[accepted,:]  # Accepted (validated) edge points
	
	# plotting
	vectors = edg-org # show the vectors also
	plt.imshow(img)
	plt.quiver(org[:,1],org[:,0],vectors[:,1],vectors[:,0],angles='xy',
		scale_units='xy', scale=1, color = 'orange')
	plt.scatter(edga[:,1],edga[:,0],c='blue',marker='.')
	plt.title("Validated Points")
	plt.show()
	
	# save
	print("Paramters:\n Camera %s:	K_s=%0.5f	k_min=%0.5f	k_max=%0.5f	l_max=%0.5f	phi_min=%0.5f outlier_sigma=%0.5f	blur=%0.5f"
		%(camera,K_s,k_min,k_max,l_max,phi_min,outlier_sigma, blur))
	save = input("save current parameters?(y/n)	")
	if save=="y":
		savename = fname_dir(fimg)
		savename = savename+os.sep+camera+"_ghosal_edge_parameters.txt"
		savematrix = np.squeeze([K_s,k_min, k_max, l_max, phi_min, outlier_sigma, blur])
		np.savetxt(savename,savematrix,delimiter="	")
		print("saved\n",savematrix)
	

############# EXTERNAL #################################################

def animate(directory,gifname,n_t,step=2,duration=0.2):
	"""
	n_t is the nuber of images you want
	step is how many images int the directory to skip each timestep
	duration is the duration of each frame in the final gif
	Based on the gif code of the SP classes of M Mendez 2019
	"""
	# create list of filenames
	fnames = dir_fname(directory,"*")
	# create list of plots
	images=[]    
	for k in range(0,n_t):
		k = k*step
		print('Mounting Im '+ str(k))
		FIG_NAME=fnames[k]
		images.append(imageio.imread(FIG_NAME)) # read
	# Now we can assemble the video
	imageio.mimsave(gifname, images,duration=duration) # create gif
	print('Animation'+gifname+'Ready')
	return True
