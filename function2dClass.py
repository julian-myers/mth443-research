"""
Base class for test functions of two variables.

"""

from abc import ABC, abstractmethod

class function2d(ABC):
	
	@abstractmethod
	def F(self,x,y): pass      # function definition
	
	@abstractmethod
	def x1(self,x,y): pass     # 1st derivative wrt x
	
	@abstractmethod
	def x2(self,x,y): pass     # 2nd derivative wrt x
	
	@abstractmethod
	def x3(self,x,y): pass     # 3rd derivative wrt x
	
	@abstractmethod
	def x4(self,x,y): pass     # 4th derivative wrt x
	
	@abstractmethod
	def y1(self,x,y): pass     # 1st derivative wrt y
	
	@abstractmethod
	def y2(self,x,y): pass     # 2nd derivative wrt y
	
	@abstractmethod
	def y3(self,x,y): pass     # 3rd derivative wrt y
	
	@abstractmethod
	def y4(self,x,y): pass     # 4th derivative wrt y
	
	@abstractmethod
	def div(self,x,y): pass    # divergence f_x + f_y
	
	@abstractmethod
	def L(self,x,y): pass      # Laplacian f_{xx} + f_{yy}
		
	@abstractmethod
	def B(self,x,y): pass      # Biharmonic operator f_{xxxx} + 2 f_{xxyy}  + f_{yyyy} 
		
	@abstractmethod
	def p12(self,x,y): pass    # mixed partial f_{xyy} 
	
	@abstractmethod
	def p21(self,x,y): pass    # mixed partial f_{xxy} 
			
	@abstractmethod
	def p22(self,x,y): pass    # mixed partial f_{xxyy} 
	
	
	
	
	
