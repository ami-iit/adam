from adam.core.link_parametric import linkParametric


class jointParametric:
    """Class for joint whose parent is parametric w.r.t. length and density"""

    def __init__(self, joint_name: str, parent_link: linkParametric, joint) -> None:
        self.jointName = joint_name
        self.parent_link_name = parent_link.name
        self.joint = joint
        self.parent_link = parent_link
        self.parent_link_offset = self.parent_link.offset
        joint_offset = self.parent_link.compute_joint_offset(joint)
        self.offset = joint_offset

    def update_joint(self):
        length = self.parent_link.get_principal_lenght_parametric()
        # Ack for avoiding depending on casadi
        vo = self.parent_link.origin[2]
        xyz = self.parent_link.zeros(3)
        xyz[0] = self.joint.origin.xyz[0]
        xyz[1] = self.joint.origin.xyz[1]
        xyz[2] = self.joint.origin.xyz[2]
        if self.joint.origin.xyz[2] < 0:
            xyz[2] = -length + self.parent_link_offset - self.offset
        else:
            xyz[2] = vo + length / 2 - self.offset
        self.xyz = xyz

    def update_parent_link_and_joint(self, length_multiplier, density):
        self.parent_link.update_link(length_multiplier, density)
        self.update_joint()
