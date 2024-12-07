"""
Provides center distributions in one, two, and three dimensions
for some commonly used domains.

suggested import: import rbfCenters as RC

--------------------- function summary ------------------------------------

1d

xGL - Chebyshev-Gauss-Lobatta (CGL) points
xKT - Kosloff/Tal-Ezer map - less densely clusted around endpoints than GCL pts
R1Points - 1d quasi-random points

2d

Hammersley2d - quasi-random points on the unit square
Halton2d - quasi-random points on the unit square
R2Points -  quasi-random points on the unit square

rectangleCenters - maps quasi-random points from the unit square to rectangles
circleCenters - maps quasi-random points from the unit square to to circles of
			    radius R. Does not put points on the boundary
circleBndCenters - adds a uniform coverage of the boundary from centers produced
	               by the function circleCenters
circleUniformCenters - uniform distributed centers on a cirlce of radius R
					   including the boundary
circlesVogelCenters - spiral pattern of centers on a circle of radius R		
annulusCenters - centers on an annulus 

3d

sphereCenters - spiral and fibonacci centers on the surface of a sphere
				   

"""

# import scipy as sp
from numpy import cos, pi, arange, arcsin, sin, zeros, sqrt, sort, meshgrid, linspace
from numpy import imag, real, angle, exp, where, array, concatenate, amin, append, arccos
import matplotlib.pyplot as pplt


# -----------------------------------------------------------------------------------------------
# -------------------- 1d centers ---------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def xGL(N,A=-1,B=1):
	""" Chebyshev-Gauss-Lobatta (CGL) points, associated with the Chebyshev
         pseudospectral method, that cluster densely around the endpoints

		input
		  N     number of centers
		 A, B   centers are on the interval [A,B]
		
		output
		  xc   CGL points on the interval [A,B]
	"""
	return  -0.5*(B-A)*cos(pi*arange(N)/(N-1)) + 0.5*(B+A)

# ------------------------------------------------------------------------------------------------

def xKT(N,A=-1,B=1,alpha=0.99):
	""" Kosloff/Tal-Ezer (KT) map.  Redistributes the CGL points
        within the interval [A,B].

		input
		  N     number of centers
		 A, B   centers are on the interval [A,B]
		alpha   map parameter 0 < alpha < 1, lim alpha -> 0  CGL 
											 lim alpha -> 1  uniform
		output
		  xc   mapped CGL points on the interval [A,B]
	"""
	return 0.5*(B-A)*arcsin(-alpha*cos(pi*arange(N)/(N-1)))/arcsin(alpha) + 0.5*(B+A)

# -----------------------------------------------------------------------------------------------



def R1Points(N,A=0,B=1,plt=False,a0=0.5):
	"""
	R1 points on (A,B), the endpoints are not included.
	
	inputs:
	   N      the number of centers
	   A, B   interval (A,B)
	   plt    logical variable for plotting
	    a0    initial point in the sequence
		
    output:
	    x     array of N centers on (A,B)
	
	
    reference: http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
	
	"""
	
	x = zeros(N)
	g = 0.5*(1 + sqrt(5))      # the Golden ratio; also the positive solution of x^2 - x - 1 = 0
	a1 = 1.0/g

	for k in range(N):
		x[k] = (a0 + a1*k)%1
		
	x = (B - A)*x + A
	x = sort(x)
		
	if plt:
		pplt.plot(x,zeros(N),'b.')
	
	return x

# -----------------------------------------------------------------------------------------------
# -------------------- 2d centers ---------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

# domain types: rectangles, circles, annular

# -----------------------------------------------------------------------------------------------
# --------------------------- rectangular -------------------------------------------------------
# -----------------------------------------------------------------------------------------------


def Hammersley2d(N, plt=False,p1=2.0):
	""" Quasi-random p1-Hammersley points on the unit square.

		inputs
		  N      number of centers
		 plt     logical variable: True -> plot centers, False -> no plot
		  p1      depends on one prime integer, default is 2
		
		output
		  (x,y)  center locations
		
		example usage:
		  1) 
	"""
	
	x, y = zeros((N,)), zeros((N,))
	p1i = 1.0/p1
	
	for k in range(N):
		u = 0
		p = p1i   
		kk = k
		while kk>0:
			if kk & 1:
				u += p
			p *= 0.5
			kk >>= 1
		v = (k + 0.5)/N
		x[k], y[k] = u, v
		
		if plt:
			pplt.scatter(x,y,c='g',marker='.')
			pplt.show()
	
	return x, y

# -------------------------------------------------------------------------------------------------


def Halton2d(N, plt=False, p1=2, p2=3):
	""" Quasi-random (p1,p2)-Halton points on the unit square.

		inputs
		  N      number of centers
		 plt     logical variable: true -> plot centers, false -> no plot
		 p1      depends on 2 prime numbers, default value for the first is p1=2 
		 p2      second prime number which is 3 by default
		
		output
		  (x,y)  center locations
  
	"""
	
	x, y = zeros((N,)), zeros((N,))
	p1i = 1.0/p1

	k = 0
	while k+1!=N:
		u = 0
		p = p1i
		kk = k
		while kk>0:
			if kk & 1:
				u += p
			p *= 0.5
			kk >>= 1
		v = 0
		ip = 1.0/p2
		p = ip
		kk = k
		while kk>0:
			a = kk % p2
			if a!=0:
				v += a*p
			p *= ip
			kk = int(kk/p2)
		x[k], y[k] = u, v
		k += 1
		
		if plt:
			pplt.scatter(x,y,c='g',marker='.')
			pplt.show()
		
	return x, y

# -----------------------------------------------------------------------------------------------



def R2Points(N,plt=False,a0=(0.5,0.5)):
	""" R2 points on [0,1] x [0,1]
	
       reference: # http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
	
	inputs:
	   N     desired number of centers
	  plt    True -> plot centers, False -> do not plot
	  a0     initial point in the sequence
	  
	outputs:
	  x, y   coordinates of the centers
	  
	"""
	x, y = zeros(N), zeros(N)

	g = 1.32471795724474602596    # solutions of x^3 - x - 1 = 0
	a1, a2 = 1.0/g, 1.0/(g**2)

	for k in range(N):
		x[k], y[k] = (a0[0] + a1*k)%1, (a0[1] + a2*k)%1
		
	if plt:
		pplt.scatter(x,y,c='g',marker='.')
		pplt.show()
		
	return x, y
	

# ------------------------------------------------------------------------------------------------


def rectangleCenters(N,a=0,b=1,c=0,d=1,ch=0,plt=False,Nx=100,Ny=100):
	""" Quasirandom centers on a rectangle [a,b] x [c,d]. 

		inputs
		   N      Number of centers in the covering square.  The number of centers
				  returned is less than N. Used for quasi-random centers
		
		  ch          0        tensor product uniform
				      1        Halton
					  2        Hammersley
					  3        R2
					  5        tensor product Gauss-Lobbato
					  6        tensor product KT mapped Gauss-Lobbato
	a,b,c,d       rectangular domain [a,b] x [c,d]
		  plt     Logical variable for plotting option
		   Nx     number of points in the x-direction (tensor product)
		   Ny     number of points in the y-direction (tensor product)
	  
	   outputs
		 x, y     center coordinates
		 
	   example usage:
	      x, y = rc.rectangleCenters(500,a=1,b=2,c=3,d=5,ch=3,plt=True,Nx=100,Ny=100)
	"""
	
	if ch==0:                      # SOMETHING WRONG HERE
		x, y = meshgrid( linspace(a,b,Nx), linspace(c,d,Ny))
	elif ch==1:
		x, y = Halton2d(N)         # (2,3)-Halton points
	elif ch==2:
		x, y = Hammersley2d(N)     # 2-Hammersly points
	elif ch==3:
		x, y = R2Points(N)
	elif ch==5:
		x, y = meshgrid( xGL(Nx,a,b), xGL(Ny,c,d) )
	elif ch==6:                    # KT map with alpha = 0.99              
		x, y = meshgrid( xKT(Nx,a,b), xKT(Nx,c,d))
		
	x, y = x.flatten(), y.flatten()
		
	if plt:
		pplt.scatter(x,y,c='g',marker='.')
		pplt.show()
		
	return x, y
	
		
		
# -----------------------------------------------------------------------------------------------
# ------------------------- circular ------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def pol2cart(th, r):
	c = r*exp( 1j*th )
	return real(c), imag(c)

def cart2pol(x,y):
	z = x + 1j*y
	return angle(z), abs(z)


def circleCenters(N,cluster=False,ch=3,R=1,center=(0,0),plt=False):
	""" Quasirandom centers on a circle of radius R. The centers
                are either based on a Hammersley or Halton sequence.
  inputs
     N      Number of centers in the covering square.  The number of centers
            returned is less than N.
   cluster  Logical variable for clustering option
    ch          1        Halton
	            2        Hammersley
				3        R2 points  (default)
				4        Vogel
     R      Radius of the circle (default of R=1)
    plt     Logical variable for plotting option (False by default)

 outputs
   x, y     center coordinates

 example usage: 1) 
	"""
	
	if ch==4:
		x, y = circlesVogelCenters(N,radius=1.0,center=(0,0),plt=False)
	elif ch==1:
		x, y = Halton2d(N)       # (2,3)-Halton points
	elif ch==2:
		x, y = Hammersley2d(N)   # 2-Hammersly points
	elif ch==3:
		x, y = R2Points(N)
	
	
	if ch != 4:
		x, y = 2*x - 1, 2*y - 1                # [0,1]^2 --> [-1,1]^2
		I = where( x**2 + y**2 <= 1 )       # restrict from square to circle
		x, y = x[I], y[I]
	
	if cluster:
		t, r = cart2pol(x,y)
		r = sin(0.5*pi*r)
		x, y = pol2cart(t,r)
			
	x, y = R*x, R*y;                      # adjust to have radius R
	x, y = x + center[0], y + center[1]
	
	if plt:
		pplt.scatter(x,y,c='g',marker='.')
		pplt.show()
	return x, y

# -----------------------------------------------------------------------------------------------



def circleBndCenters(xi,yi,R=1,Nb=70,tol=0.01,plt=False):
	"""  Adds a uniform covering to the boundary of a circle 
     of radius R that is covered on the interior by quasi-random centers.
     Useful in the discretization of PDEs in order to inforce BCs.

		inputs
		  xi, yi  scattered center locations on a circle of radius R
			 R    radius of the circle
			Nb    number of centers to be placed equally spaced on the boundary 
					  of the circle
		   tol    delete any interior centers that are within distance tol
					  of a boundary center
		   plt    logical variable, plot the centers
		
		outputs
		  x, y   interior and boundary centers with the boundary centers having
					indicies 0 to Nb-1
		
		 example usage:
		 xi, yi = rc.circleCenters(500,cluster=False,ch=2,R=1,plt=False)
         x, y = rc.circleBndCenters(xi,yi,R=1,Nb=70,tol=0.01,plt=True)
	"""
	t = linspace(0, 2*pi, Nb + 1)
	t = t[:Nb]
	xb, yb = R*cos(t), R*sin(t)

	rd, rx, ry = rbfx.distanceMatrix2d(xi,yi,xb,yb) # distance between boundary and interior points
	m = amin(rd,axis=0)                          # find the min in each column
	xi, yi = xi[m>tol], yi[m>tol]                   # get rid of interior centers too close to a bnd center
	x, y = append(xb,xi), append(yb,yi)       # merge interior and boundary centers
		
	if plt:
		pplt.scatter(x,y,c='g',marker='.')
		pplt.scatter(x[:Nb],y[:Nb],c='b',marker='.')
		pplt.show()
		
	return x, y

	
	
# -------------------------------------------------------------------------------------------------

def theta(k): return linspace(0, 2*pi, k+1)[0:-1]
def shift(t0): return t0 + ( t0[1] - t0[0] )/2.0
	
def circleUniformCenters(N,fixed=False,radius=1.0,center=(0,0),plt=False,verbose=False):
	""" Uniformly space centers on a circle.

   inputs 
     N      The number of centers returned
radius      Radius of the circle
    plt     Logical variable for plotting option
 center	    circle centered at(x0,y0)
 fixed      if fixed = True, N centers are returned
               fixed = False, centers are added so that the outter
                      2 circles have the same number of centers

 outputs
   x, y     center coordinates
     Nb     The number of center located on the boundary which are in
            the last Nb locations of the returned vector.  Useful for
            enforcing PDE boundary conditions. 

example usage:
	x, y, Nb = rc.circleUniformCenters(500,fixed=True,radius=2.0,center=(3,2),plt=True,verbose=True)

	"""
	x, y = array([0]), array([0])
	m = int(round(sqrt(pi + 4*(N-1)) - sqrt(pi))/(2*sqrt(pi)))   # number of circles
	dd = pi*(m+1)/(N-1)
	for i in arange(1,m):
		ri = float(i)/m
		ni = round(2*pi*ri/dd)   # number of points on circle
		t = theta(ni)
		if i%2==0: t = shift(t)  # stagger angle of alternating circles
		X, Y = ri*sin(t), ri*cos(t)
		x, y = concatenate((x,X)), concatenate((y,Y))

	if fixed:
		t = theta(N - len(x))
	else:
		t = theta(ni)
		
	if m%2==0: t = shift(t)   # add last circle of r=1
	Nb = len(t)       # number of boundary centers
	x, y = concatenate((x,sin(t))), concatenate((y,cos(t)))
	x, y = radius*x, radius*y
	
	if verbose:
		print('N = {:d}'.format(N))
		print('boundary centers = {:d}'.format(Nb))
		print('total centers = {:d}'.format(len(x)))
		print('number of circles = {:d}'.format(m))

	x, y = x + center[0], y + center[1]
	if plt:
		pplt.scatter(x,y,c='g',marker='.')
		pplt.scatter(x[-Nb:],y[-Nb:],c='k',marker='.')
		pplt.show()
	return x, y, Nb



def circlesVogelCenters(N,radius=1.0,center=(0,0),plt=False):
	"""
	Produces a spiral type pattern of centers on a circle.
	
	inputs:
	   N         the number of centers
	   radius    radius of the circle
	   center    center of the circle
	   plt       logical variable for plotting
	   
	output:  N centers (x, y)
	
	reference:
	  http://blog.marmakoide.org/?p=1
	"""
	k = arange(1,N+1)
	th = pi*(3 - sqrt(5))
	fac = sqrt(k/N)
	x, y = fac*cos(k*th), fac*sin(k*th)
	x, y = radius*x, radius*y
	x, y = x + center[0], y + center[1]
	if plt:
		pplt.scatter(x,y,c='g',marker='.')
		pplt.show()
	return x, y

# -----------------------------------------------------------------------------------------------
#------------------- annular --------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------


def annulusCenters(N,dist=0,ri=0.25,ro=1,nbo=200,nbi=100,tol=0.01,plt=False): #,cluster=False):
	""" center distributions on annular domains
	      
	    dist    0        uniform
	            1        Halton
	            2        Hammersley
				3        R2 points  (default)
				4        Vogel
	
	"""
	
	fac = 1.25
	if dist == 0:
		x, y, _ = circleUniformCenters(N,False,ro - fac*tol,(0,0),False,False)
	elif dist == 1:
		x, y = circleCenters(N,cluster=False,ch=1,R=ro - fac*tol,plt=False)
	elif dist == 2:
		x, y = circleCenters(N,cluster=False,ch=2,R=ro - fac*tol,plt=False)
	elif dist == 3:
		x, y = circleCenters(N,cluster=False,ch=3,R=ro - fac*tol,plt=False)
	elif dist == 4:
		x, y = circlesVogelCenters(N,radius=ro - fac*tol,center=(0,0),plt=False)
	
	I = where( sp.sqrt(x**2 + y**2) > ri )  
	x, y = x[I], y[I]

	nbt = nbo + nbi
	x, y = circleBndCenters(x,y,ro,nbo,tol)
	x, y = circleBndCenters(x,y,ri,nbi,tol)
	
	# if cluster:
	# 	t, r = cart2pol(x,y)
	# 	r = 36*r**2 - 24*r + 4.0
	# 	x, y = pol2cart(t,r)
	
	if plt:
		pplt.scatter(x,y,c='g',marker='.')
		pplt.scatter(x[:nbt],y[:nbt],c='k',marker='.')
		pplt.show()
		pplt.axis('equal')
	
	return x, y
	
# -----------------------------------------------------------------------------------------------	
# ------------------------------------------ 3d -------------------------------------------------
# -----------------------------------------------------------------------------------------------

# WORK IN PROGRESS, see for example
# https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere/26127012#26127012

# -------------------------------------------------------

import math as mt
import random as rn

def fibonacciSphere(N,randomize=True):
    rnd = 1.0
    if randomize:
        rnd = rn.random()*N

    x, y, z = [], [], []
    offset = 2.0/N
    increment = mt.pi * (3.0 - mt.sqrt(5.0))

    for i in range(N):
        yt = ((i*offset) - 1) + (0.5*offset)
        y.append( yt )
        r = mt.sqrt(1 - pow(yt,2))
        phi = ((i + rnd) % N)*increment
        x.append( mt.cos(phi)*r )
        z.append( mt.sin(phi)*r )
        
    return sp.array(x), sp.array(y), sp.array(z)
    
# ----------------------------------------------------------


import mpl_toolkits.mplot3d

def spiralSphere(N):
	"""
	Spiral type covering of the surface of a unit sphere.
	
	reference: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere/44164075#44164075
	
	"""
	indices = arange(0, N, dtype=float) + 0.5
	phi = arccos(1 - 2*indices/N)
	theta = pi * (1 + 5**0.5)*indices
	x, y, z = cos(theta)*sin(phi), sin(theta)*sin(phi), cos(phi)
	return x, y, z

# ------------------------------
	

def sphereCenters(N,ch=0,R=1,plt=False,randomize=False):
	""" Centers on the surface of a sphere.
	
  inputs:
     N      Number of centers on the surface of a sphere or radius R 
            
    ch          0         spiral    
            otherswise    Fibonacci
			
     R      Radius of the sphere (default of R=1)
    plt     Logical variable for plotting option (False by default)

 outputs
   x, y, z    center coordinates

 example usage:
   1) examples/interp3d.py
   
	"""
	
	if ch==0:
		x, y, z = spiralSphere(N)
	else:
		x, y, z = fibonacciSphere(N,randomize)
		
	if R!=1: x, y, z = R*x, R*y, R*z
		
	if plt:
		pplt.figure().add_subplot(111, projection='3d').scatter(x, y, z)
		pplt.show()
		
	return x, y, z
		
		
	
	
