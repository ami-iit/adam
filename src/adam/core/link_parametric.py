from enum import IntEnum
import math

class I_parametric():
    def __init__(self) -> None:
        self.ixx = 0.0
        self.ixy = 0.0
        self.ixz = 0.0
        self.iyy = 0.0
        self.iyz = 0.0
        self.izz = 0.0

class Geometry(IntEnum):
    """The different types of geometries that constitute the URDF"""
    BOX = 1
    CYLINDER = 2
    SPHERE = 3

class GrowingDirection(IntEnum):
    """The possible sides of a box geometry"""
    X = 0
    Y = 1
    Z = 2

class linkParametric():
    def __init__(self, link_name: str, link, R,index, growing_direction:GrowingDirection = GrowingDirection.Z) -> None:
        self.name = link_name
        self.link = link  
        self.R = R
        self.geometry_type, self.visual_data = self.get_geometry(self.get_visual())
        self.offset = self.compute_offset()
        self.growing_direction = growing_direction 
        self.index = index        

    def set_external_methods(self, zeros, fk): 
        self.zeros = zeros
        self.fk = fk 

    def update_link(self,lenght_multiplier_vector, density_vector):
        lenght_multiplier_i = lenght_multiplier_vector[self.index,:]
        density_i = density_vector[self.index]
        self.update_dimensions(lenght_multiplier_i)
        self.volume = self.compute_volume()
        self.mass = self.compute_mass(density_i)
        self.origin = self.modify_origin(lenght_multiplier_i)
        self.I = self.compute_inertia_parametric()
        self.inertial = self.I

    def update_dimensions(self, length_multiplier): 
        if self.geometry_type == Geometry.BOX:
            visual_data_new =[0.0, 0.0, 0.0]
            visual_data_new[0] = self.visual_data.size[0] * length_multiplier[0]
            visual_data_new[1] = self.visual_data.size[1] * length_multiplier[1]
            visual_data_new[2] = self.visual_data.size[2] * length_multiplier[2]
        elif self.geometry_type == Geometry.CYLINDER:
            visual_data_new = [0.0, 0.0]
            visual_data_new[0] = self.visual_data.length * length_multiplier[0]
            visual_data_new[1] = self.visual_data.radius * length_multiplier[1]
        elif self.geometry_type == Geometry.SPHERE:
            visual_data_new = 0.0
            visual_data_new = self.visual_data.radius * length_multiplier[0]
        self.visual_data_new = visual_data_new

    def get_principal_lenght(self): 
        visual = self.get_visual()
        xyz_rpy = [*visual.origin.xyz, *visual.origin.rpy] 
        if self.geometry_type == Geometry.CYLINDER:
            if(xyz_rpy[3] < 0.0 or xyz_rpy[4] > 0.0):
                v_l = 2*self.visual_data.radius # returning the diameter, since the orientation of the shape is such that the radius is the principal lenght 
            else: 
                v_l=self.visual_data.length # returning the lenght, since the orientation of the shape is such that the radius is the principal lenght 
        elif(self.geometry_type == Geometry.SPHERE): 
            v_l = self.visual_data.radius
        elif(self.geometry_type == Geometry.BOX): 
            v_l= self.visual_data.size[2]
        else:
            raise Exception(f"THE GEOMETRY IS NOT SPECIFIED")
        return v_l 

    def get_principal_lenght_parametric(self): 
        visual = self.get_visual()
        xyz_rpy = [*visual.origin.xyz, *visual.origin.rpy] 
        length_vector = self.zeros(3)
        if self.geometry_type == Geometry.CYLINDER:
            length_vector[0] = 2*self.visual_data_new[1]
            length_vector[1] = 2*self.visual_data_new[1]
            length_vector[2] = self.visual_data_new[2]        
        elif(self.geometry_type == Geometry.SPHERE): 
            length_vector[0] = 2*self.visual_data_new
            length_vector[1] = 2*self.visual_data_new
            length_vector[2] = self.visual_data_new
        elif(self.geometry_type == Geometry.BOX): 
            length_vector[0] = self.visual_data_new[0]
            length_vector[1] = self.visual_data_new[1]
            length_vector[2] = self.visual_data_new[2]
        else:
            raise Exception(f"THE GEOMETRY IS NOT SPECIFIED")
        length_vector_rotate = self.R@length_vector.array
        v_l= length_vector_rotate[self.growing_direction]
        return v_l 
   
    def compute_offset(self): 
        visual = self.get_visual()
        xyz_rpy = [*visual.origin.xyz, *visual.origin.rpy] 
        v_l=  self.get_principal_lenght()
        v_o = xyz_rpy[2]
        if(v_o<0):
            link_offset = v_l/2 + v_o
        else:
            link_offset = (v_o - v_l/2)
        return link_offset

    def compute_joint_offset(self,joint_i): 
         # Taking the principal direction i.e. the length 
        visual = self.get_visual()
        xyz_rpy = [*visual.origin.xyz, *visual.origin.rpy] 
        v_l= self.get_principal_lenght()
        j_0 = joint_i.origin.xyz[2]
        v_o =  xyz_rpy[2]
        if(j_0<0):
            joint_offset_temp = -(v_l + j_0-self.offset)
            joint_offset = joint_offset_temp
        else:
            joint_offset_temp = v_l + self.offset - j_0
            joint_offset = joint_offset_temp
        return joint_offset
    
    def get_visual(self):
        """Returns the visual object of a link"""
        return self.link.visuals[0]

    @staticmethod
    def get_geometry(visual_obj):
        if hasattr(visual_obj.geometry, "size"):
            return [Geometry.BOX, visual_obj.geometry]
        elif hasattr(visual_obj.geometry, "length"):
            return [Geometry.CYLINDER, visual_obj.geometry]
        elif hasattr(visual_obj.geometry,"radius"):
            return [Geometry.SPHERE, visual_obj.geometry]

    """Function that starting from a multiplier and link visual characteristics computes the link volume"""
    def compute_volume(self):
        volume = 0.0
        """Modifies a link's volume by a given multiplier, in a manner that is logical with the link's geometry"""
        if self.geometry_type == Geometry.BOX:
            volume = self.visual_data_new[0] * self.visual_data_new[1] * self.visual_data_new[2]
        elif self.geometry_type == Geometry.CYLINDER:
            volume = math.pi * self.visual_data_new[1] ** 2 * self.visual_data_new[0]
        elif self.geometry_type == Geometry.SPHERE:
            volume = 4 * math.pi * self.visual_data_new ** 3 / 3
        return volume

    """Function that computes the mass starting from the density, the length multiplier and the link"""
    def compute_mass(self, density):
        """Changes the mass of a link by preserving a given density."""
        mass = 0.0
        mass = self.volume * density
        return mass

    def modify_origin(self, lenght_multiplier):
        origin = [0.0,0.0,0.0,0.0,0.0,0.0]
        visual = self.get_visual()
        """Modifies the position of the origin by a given amount"""
        xyz_rpy = [*visual.origin.xyz, *visual.origin.rpy] 
        v_o = visual.origin.xyz[2] 
        length = self.get_principal_lenght_parametric()
        origin = xyz_rpy
        if(v_o<0):
            origin[self.growing_direction] = self.offset-length/2
        else:
            origin[self.growing_direction] = length/2 +self.offset
        if self.geometry_type == Geometry.SPHERE:
            origin = xyz_rpy
        return origin

    def compute_inertia_parametric(self):
        I = I_parametric()
        
        visual = self.get_visual() 
        xyz_rpy =  [*visual.origin.xyz, *visual.origin.rpy]
        """Calculates inertia (ixx, iyy and izz) with the formula that corresponds to the geometry
        Formulas retrieved from https://en.wikipedia.org/wiki/List_of_moments_of_inertia"""
        if self.geometry_type == Geometry.BOX:
            I.ixx = self.mass * (self.visual_data_new[1] ** 2+ self.visual_data_new[2] ** 2)/12
            I.iyy = self.mass* (self.visual_data_new[0]**2 + self.visual_data_new[2]**2)/12
            I.izz = self.mass * (self.visual_data_new[0]**2 + self.visual_data_new[1]**2)/12
        elif self.geometry_type == Geometry.CYLINDER:
            i_xy_incomplete = (
                3 *(self.visual_data_new[1] ** 2) + self.visual_data_new[0] ** 2
            ) / 12
            I.ixx = self.mass * i_xy_incomplete
            I.iyy = self.mass * i_xy_incomplete
            I.izz = self.mass * self.visual_data_new[1] ** 2 / 2
            return I
        elif self.geometry_type == Geometry.SPHERE:
            I.ixx = 2 * self.mass * self.visual_data_new** 2 / 5
            I.iyy = I.ixx
            I.izz = I.ixx
        return I

class jointParametric:
    def __init__(self, joint_name:str, parent_link:linkParametric, joint) -> None:
        self.jointName = joint_name
        self.parent_link_name = parent_link.name
        self.joint = joint
        self.parent_link = parent_link
        self.parent_link_offset = self.parent_link.offset
        joint_offset = self.parent_link.compute_joint_offset(joint)
        self.offset = joint_offset
        
    def modify(self, length_multiplier):
        self.parent_link.update_dimensions(length_multiplier)
        length = self.parent_link.get_principal_lenght_parametric()
        # Ack for avoiding depending on casadi 
        vo = self.parent_link.origin[2]
        xyz =self.parent_link.zeros(3)
        xyz[0] = self.joint.origin.xyz[0]
        xyz[1] = self.joint.origin.xyz[1]
        xyz[2] = self.joint.origin.xyz[2]
        # xyz_rpy_temp=  [*self.joint.origin.xyz, *self.joint.origin.rpy]
        # xyz_rpy[3] = xyz_rpy_temp[3]
        # xyz_rpy[4] = xyz_rpy_temp[4]
        # xyz_rpy[5] = xyz_rpy_temp[5]
        if(self.joint.origin.xyz[2]<0): 
            xyz[2] = -length +self.parent_link_offset - self.offset   
        else:
            xyz[2] = vo+ length/2 - self.offset
        self.xyz = xyz.array