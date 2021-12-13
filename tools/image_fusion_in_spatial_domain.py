"""
    创建时间：2018年07月31日13:47:21
    这个脚本实现对 3D 图像进行空域融合，使得融合后的神经元 3D 图像中包含一个完整的神经元，和一些具有干扰作用的神经纤维（即不完整的神经元）
    我们的最终目的是构建这样的训练数据：3D 神经元图像中包含一个完整的神经元、一些干扰这个完整神经元的神经纤维
                  以及这样的标签数据：3D 标签中，既不处在完整神经元也不处在干扰神经纤维的相应坐标（即背景区域坐标）标为0、
                                            完整的神经元的相应坐标标为1、
                                            干扰这个完整神经元的神经纤维的相应坐标标为2
    之所以构建这样数据，是基于以下现状：
        目前已经有比较成熟的神经元追踪算法，如 APP1、APP2、SmartTracing等，这些算法对这样的神经元图像的追踪效果较好 —— 图像中只包含一个神经元或几个相互分离的神经元、图像较清晰或经过较好的去噪
        BigNeuron 是一个神经元追踪研究项目，提供了如APP1、APP2、SmartTracing 等算法，以及大量的神经元图像数据，如前所述，这些图像数据或是只包含一个神经元或是包含几个相互分离的神经元
        但我们需要处理的神经元图像是对脑样本切片进行高尔基染色法得到的图像，其中包含大量相互交织在一起的神经元，无法直接基于现有的追踪算法将这些神经元追踪提取出来，我们目前的思路如下：
            1. 将其中一个完整神经元从大图像中裁剪出来，这样裁剪得到的3D图像中包含大量其他神经元的部分纤维
            2. 从这个3D图像中将提取出其所包含的完整神经元
            3. 使用诸如 APP1、APP2、SmartTracing等算法对所提取到的完整神经元进行追踪
        关键在第二步，若进行人工提取，将花费大量时间（一个完整的小鼠脑中大约包含500万个神经元），我们希望使用如深度学习等人工只能的方法进行提取，这需要大量的训练数据
        但显然也缺少这样的标注后的训练数据，因此对BigNeuron中的神经元图像数据进行以上方式的融合处理，构建训练数据
    作者：李秋富
"""
import os
import cv2
import numpy as np
from tools.math_tool import get_connect_coordinate_3d, rotate_3d, getshape_rotate_3d, distance, rotate_3d_inverse
from tools.math_tool import get_connect_coordinate_3d_new
from collections import OrderedDict
from tools.image_3D_io import save_image_3d
import copy
import math


class NeuronNode():
    """
    这个类描述神经元节点，主要包括 编号、类型、三维坐标、半径、父节点等属性；注意这里节点的坐标可能不为整数
    """
    def __init__(self):
        self.id = None
        self.type = None
        self.x = None
        self.y = None
        self.z = None
        self.radius = None
        self.p_id = None
        self.child_id = list()            #初始化为空，在生成神经元节点列表过程中被赋值，即被真正初始化
        self.processed = None           #表示当前节点是否被处理，若被处理则这个值被重置为1；
                                        #即便如此，并不表示该节点已经被处理完，只有当该节点以及该节点的所有子节点都被处理了才表示这个节点被处理完了，
                                        #此时可从NeurongNode_List类型中删除这个节点
        self.resolution = None

    def __str__(self):
        return ' '.join([str(self.id),
                         str(self.type),
                         str(self.x),
                         str(self.y),
                         str(self.z),
                         str(self.radius),
                         str(self.p_id),
                         str(self.child_id),
                         str(self.processed)])

    def get_around_points(self, r_reset = None, shape = None):
        """
        获取以节点坐标为中心点，self.radius为半径内的左右坐标点
        :param r_reset: 有些情况下要对该节点的半径进行重置
        :return:
        """
        x = round(self.x)
        y = round(self.y)
        z = round(self.z)
        r = round(self.radius) if r_reset == None else round(r_reset)
        r_x = max(round(r / self.resolution[0]), 1)
        r_y = max(round(r / self.resolution[1]), 1)
        r_z = max(round(r / self.resolution[2]), 1)
        if shape == None:
            return [(zz, yy, xx)
                    for xx in range(x - r_x, x + r_x + 1)
                    for yy in range(y - r_y, y + r_y + 1)
                    for zz in range(z - r_z, z + r_z + 1)
                    if (((xx - x)/r_x) ** 2 + ((yy - y)/r_y) ** 2 + ((zz - z)/r_z) ** 2) <= 1]
        elif len(shape) != 3:
            raise ValueError('形状参数有误')
        else:
            depth = shape[0]
            height = shape[1]
            width = shape[2]
            return [(zz, yy, xx)
                    for xx in range(x - r_x, x + r_x + 1)
                    for yy in range(y - r_y, y + r_y + 1)
                    for zz in range(z - r_z, z + r_z + 1)
                    if ((((xx - x)/r_x) ** 2 + ((yy - y)/r_y) ** 2 + ((zz - z)/r_z) ** 2) <= 1
                        and (0 <= zz and zz < depth)
                        and (0 <= yy and yy < height)
                        and (0 <= xx and xx < width))]

    def get_connect_points(self, another_node, shape = None, r_offset = 2):
        """
        获取该节点与其某个子节点连线之间的坐标点
        :param another_node: NeuronNode类型
        :return:
        """
        assert isinstance(another_node, NeuronNode)
        if another_node.id not in self.child_id:
            return []
        assert another_node.id in self.child_id
        if self.radius == 0 or another_node.radius == 0:
            points_list = []
        else:
            points_list =  get_connect_coordinate_3d_new((round(self.x), round(self.y), round(self.z)),
                                                     (round(another_node.x), round(another_node.y), round(another_node.z)),
                                                     radius = min(self.radius, another_node.radius) + r_offset,
                                                         resolution = self.resolution)
        if shape == None:
            return [(zz,yy,xx) for (xx,yy,zz) in points_list]
        elif len(shape) != 3:
            raise ValueError('形状参数有误')
        else:
            return [(zz,yy,xx) for (xx,yy,zz) in points_list
                    if (zz >=0 and zz < shape[0]
                        and yy >= 0 and yy < shape[1]
                        and xx >= 0 and xx < shape[2])]

    def distance_with_another(self, another_node):
        """
        检查当前节点与另一个节点的距离关系
        :param another_node:
        :return: (d0,d1,d2,d3)
                d0: 两个节点坐标之间的距离, 0表示无交集，1表示有交集
                d1: 两个节点是否有交集（在考虑半径的情况下）
                (d2,d3): 两个节点的包含情况，(0,0)表示彼此不包含，
                                          (0,1)表示当前节点被节点 another_node 包含
                                          (1,0)表示当前节点包含节点 another_node
                                          (1,1)表示当前节点与节点 another_node 几乎重合
        """
        assert isinstance(another_node, NeuronNode)
        d0 = distance((self.x,self.y, self.z), (another_node.x,another_node.y,another_node.z))
        d1 = 1 if d0 < (self.radius + another_node.radius) else 0
        d2 = (self.radius + another_node.radius - d0) / self.radius if d1 else 0
        d3 = (self.radius + another_node.radius - d0) / another_node.radius if d1 else 0
        return d0, d1, d2, d3


class NeuronNodeList():
    """
    保存多个神经节点的有序字典，字典的键是相应神经节点的编号
    """
    def __init__(self):
        """
        通常每个神经元节点列表都对应着一个神经元三维图像，这里self.width、self.height、self.depth分别是这个三维图像的宽、高、深
        """
        self._neuron_node_list = OrderedDict()
        self.width = None               #神经元所处立方体（对应着相应的三维图像数据）的尺寸，在不进行相关操作时候，可不设置
        self.height = None
        self.depth = None
        self._flag = self.flag()
        self.label_3d = None        #对应的标签文件，默认为None，由generate_label_3d生成
        self.resolution = None

    def shape(self):
        return self.depth, self.height, self.width

    def set_shape(self, shape):
        assert len(shape) == 3
        self.depth, self.height, self.width = shape

    def __len__(self):
        return self._neuron_node_list.__len__()

    def keys(self):
        return self._neuron_node_list.keys()

    def save(self, saved_file_name = None):
        """
        将某个神经元节点列表保存为 swc 文件的格式
        :param saved_file_name:
        :return:
        """
        assert saved_file_name != None
        assert self.height != None
        file = open(saved_file_name, 'w')
        for key in self._neuron_node_list.keys():
            neuron_node = self._neuron_node_list[key]
            line = '{} {} {:2f} {:.2f} {:.2f} {:.2f} {}\n'.format(neuron_node.id,
                                                                neuron_node.type,
                                                                neuron_node.x,
                                                                self.height - neuron_node.y - 1,
                                                                neuron_node.z,
                                                                neuron_node.radius,
                                                                neuron_node.p_id)
            file.write(line)
        file.close()

    def get_size_info(self):
        """
        获取当前神经元节点列表在空间中的外接立方体的尺寸，以及该立方体在完整三维神经元图像中的左下角坐标
        :return:
        """
        X = [self._neuron_node_list[key].x for key in self._neuron_node_list.keys()]
        Y = [self._neuron_node_list[key].y for key in self._neuron_node_list.keys()]
        Z = [self._neuron_node_list[key].z for key in self._neuron_node_list.keys()]
        x0 = round(min(X)) - 1
        y0 = round(min(Y)) - 1
        z0 = round(min(Z)) - 1
        width = round(max(X)) - x0 + 1
        height = round(max(Y)) - y0 + 1
        depth = round(max(Z)) - z0 + 1
        return (depth, height, width), (x0, y0, z0)

    def check_size_info(self, shape = None, coor = (0,0,0)):
        """
        检查神经元节点列表尺寸和设定的尺寸是否匹配
        :param shape: 默认是设定的尺寸 depth(Z), height(Y), width(X)
        :param coor: 默认是坐标原点 X(width), Y(height), Z(depth)
        :return:
        """
        assert isinstance(self.depth, int)
        assert isinstance(self.height,int)
        assert isinstance(self.width,int)
        shape = (self.depth, self.height, self.width) if shape == None else shape
        shape_, coor_ = self.get_size_info()
        assert shape[0] >= shape_[0]
        assert shape[1] >= shape_[1]
        assert shape[2] >= shape_[2]
        assert coor[0] <= coor_[0]
        assert coor[1] <= coor_[1]
        assert coor[2] <= coor_[2]

    def refresh_childID(self):
        """
        根据self._neuron_node_list中已有的神经节点，更新每个节点的属性值 child_id
        :param neuron_node:
        :return:
        """
        keys = self._neuron_node_list.keys()
        for key in keys:
            self._neuron_node_list[key].processed = 0
            self._neuron_node_list[key].child_id = list()
        neuronnodelist = NeuronNodeList()
        for index, key in enumerate(keys):
            neuron_node = copy.deepcopy(self._neuron_node_list[key])
            neuronnodelist.add_neuron_node(neuron_node)
        self._neuron_node_list = copy.copy(neuronnodelist._neuron_node_list)

    def flag(self):
        """
        如果swc文件中没有给出所有神经节点的半径（即所有神经节点的半径均被设置为1），则返回True，否则返回False
        :return:
        """
        sum_r = 0
        for key in self._neuron_node_list:
            sum_r += self._neuron_node_list[key].radius
        return True if sum_r == len(self._neuron_node_list) else False

    def add_neuron_node(self, neuron_node):
        """
        往有序字典中添加一个神经元节点
        :param neuron_node:
        :return:
        """
        assert isinstance(neuron_node, NeuronNode)
        if neuron_node.id in self._neuron_node_list.keys():
            return
        else:
            self._neuron_node_list[neuron_node.id] = neuron_node
            if neuron_node.p_id in self._neuron_node_list.keys() and neuron_node.id not in self._neuron_node_list[neuron_node.p_id].child_id:
                self._neuron_node_list[neuron_node.p_id].child_id.append(neuron_node.id)

    def change_other_id(self, other_node_list):
        """
        根据当前节点列表长度，修改另一个节点列表的编号和父节点编号
        :param another_noiselist:
        :return:
        """
        another_node_list = copy.deepcopy(other_node_list)
        assert isinstance(another_node_list, NeuronNodeList)
        length = self.__len__() + 1
        keys = other_node_list.keys()
        for index, key in enumerate(keys):
            node = another_node_list._neuron_node_list[key]
            id_new = index + length
            for iden in node.child_id:
                node_child = another_node_list._neuron_node_list[iden]
                node_child.p_id = str(id_new)
                node_child.processed = 1
            if (node.p_id not in keys) and (node.processed == 0):
                node.p_id = '-1'
            node.id = str(id_new)
            #print(node)
        another_node_list.refresh_childID()
        return another_node_list

    def concatenate(self, another_nodelist):
        """
        将当前 NeuronNodeList 和另一个该类型的对象连接起来，只是将被处理对象的节点列表加入当前节点列表中，不考虑其他参数
        :param another_nodelist:
        :return:
        """
        assert isinstance(another_nodelist, NeuronNodeList)
        another_nodelist = self.change_other_id(another_nodelist)
        #print(self.__len__())
        for index, key in enumerate(another_nodelist.keys()):
            node = copy.copy(another_nodelist._neuron_node_list[key])
            self.add_neuron_node(neuron_node = node)
        return self

    def distance_with_neuronnode(self, neuronnode):
        """
        检查当前节点列表与另一个节点的距离关系
        :param neuronnode:
        :return: d0,d1,d2,d3
        """
        assert isinstance(neuronnode, NeuronNode)
        d0_, d1_, d2_, d3_ = 10000, 0, 0, 0
        for key in self._neuron_node_list.keys():
            node = self._neuron_node_list[key]
            d0, d1, d2, d3 = node.distance_with_another(neuronnode)
            d0_ = d0 if d0 < d0_ else d0_
            d1_ = d1 if d1 > d1_ else d1_
            d2_ = d2 if d2 > d2_ else d2_
            d3_ = d3 if d3 > d3_ else d3_
        return d0_, d1_, d2_, d3_

    def change_size_info(self, shape = None, coor = None):
        """
        修改当前神经元节点列表在空间中的外接立方体的尺寸，以及该立方体在完整三维神经元图像中的左下角坐标
        :param shape: 将神经元节点列表所处立方体尺寸修改为此， depth(Z), height(Y), width(X)
        :param coor: 将神经元节点列表所处立方体的原点坐标平移至此， X(width), Y(height), Z(depth)，
                      这里移动的是坐标原点，相当于对原图像进行裁剪
        :return:
        """
        if shape == None and coor == None:
            raise ValueError('没有制定参数值')
        shape = list(shape)
        assert len(shape) == 3 and len(coor) == 3
        shape[0] = self.depth - coor[2] if coor[2] + shape[0] >= self.depth else shape[0]
        shape[1] = self.height - coor[1] if coor[1] + shape[1] >= self.height else shape[1]
        shape[2] = self.width - coor[0] if coor[0] + shape[2] >= self.width else shape[2]
        if shape:
            self.depth, self.height, self.width = shape
        if coor:
            for key in self._neuron_node_list.keys():
                self._neuron_node_list[key].x -= coor[0]
                self._neuron_node_list[key].y -= coor[1]
                self._neuron_node_list[key].z -= coor[2]
        return self

    def cut_whole(self, redundancy = (5,10,10)):
        """
        将神经元完整裁剪，实际上是做了一次平移，这个操作会改变当前神经元节点列表
        :param redundancy: 在完整裁剪时候，在三个轴向上保留的冗余，以使得神经元不是刚刚好贴在裁剪后的图像边儿上的
                            这里需要注意，轴顺序是 depth(Z), height(Y), width(X), 形状shape的三个参数顺序与此相同
                                        坐标顺序是 X(width), Y(height), Z(depth)
                            因此这里三元组redundancy的顺序为depth(Z), height(Y), width(X)
        :return:
        """
        shape, coor = self.get_size_info()
        coor = [x - y for (x,y) in zip(coor, redundancy[-1::-1])]
        coor = [x if x > 0 else 0 for x in coor]
        shape = [x + 2 * y for (x,y) in zip(shape, redundancy)]
        self.change_size_info(shape=shape, coor = coor)
        return self, coor

    def cut_as_swc(self, id = None, redundancy = (0,0,0)):
        """
        从神经元中裁剪出以编号为id的神经节点为父节点的所有子节点和所有间接子节点
        生成一个新的神经元节点列表，当前神经元节点列表不改变
        :param id: str(int)，节点编号
        :param redundancy: 在完整裁剪时候，在三个轴向上保留的冗余，以使得神经元不是刚刚好贴在裁剪后的图像边儿上的
                            这里需要注意，轴顺序是 depth(Z), height(Y), width(X), 形状shape的三个参数顺序与此相同
                                        坐标顺序是 X(width), Y(height), Z(depth)
                            因此这里三元组redundancy的顺序为depth(Z), height(Y), width(X)
        :return:
        """
        if id == None:
            raise ValueError('请制定节点编号')
        assert (isinstance(id, str) or isinstance(id, int))
        id = str(id) if type(id) == int else id
        assert id in self._neuron_node_list.keys()
        ids = [id]
        ids_ = ids[:]
        neuron_node_list_temp = NeuronNodeList()
        neuron_node_list_temp.set_shape(shape=self.shape())
        while ids_ != []:
            for id_ in ids:
                node = copy.copy(self._neuron_node_list[id_])
                neuron_node_list_temp.add_neuron_node(node)
                ids_.remove(id_)
                for id__ in node.child_id:
                    ids_.append(id__)
            ids = ids_[:]
        shape, coor = neuron_node_list_temp.get_size_info()
        coor = [x - y for (x,y) in zip(coor, redundancy[-1::-1])]
        coor = [x if x > 0 else 0 for x in coor]
        shape = [x + 2 * y for (x,y) in zip(shape, redundancy)]
        neuron_node_list_temp.change_size_info(shape = shape, coor = coor)
        neuron_node_list_temp.refresh_childID()
        self.refresh_childID()
        return neuron_node_list_temp, coor

    def cut_as_shape(self, shape = None, coor = None):
        """
        将当前神经元节点列表所处立方体的从 coor 开始形如 shape 内的神经元节点裁剪出来，形成新的神经元节点列表
        :param shape:
        :param coor:
        :return:
        """
        neuron_node_list_temp = NeuronNodeList()
        neuron_node_list_temp.set_shape(shape = self.shape())
        x_0 = coor[0]
        y_0 = coor[1]
        z_0 = coor[2]
        depth = shape[0]
        height = shape[1]
        width = shape[2]
        for key in self._neuron_node_list.keys():
            node = copy.copy(self._neuron_node_list[key])
            if ((node.x >= x_0 and node.x < x_0 + width)
                    and (node.y >= y_0 and node.y < y_0 + height)
                    and (node.z >= z_0 and node.z < z_0 + depth)):
                neuron_node_list_temp.add_neuron_node(node)
        neuron_node_list_temp.refresh_childID()
        neuron_node_list_temp.change_size_info(shape = shape, coor=coor)
        return neuron_node_list_temp, coor

    def rotate(self, angle_Z = 0, angle_Y = 0, angle_X = 0, point0 = None):
        """
        将当前神经节点列表中的每个节点的坐标进行旋转，当前神经节点列表的数据被改变
        :param node_list:
        :param angle_Z: 弧度值，绕 Z 轴的旋转角
        :param angle_Y: 弧度值，向 Y 轴方向的旋转角
        :param angle_X: 弧度值，向 Y 轴方向的旋转角
        :param point0: 以这个点为原点
        :return:
        """
        width, height, depth = getshape_rotate_3d((self.width, self.height, self.depth),
                                                  angle_Z = angle_Z, angle_Y = angle_Y, angle_X = angle_X)
        if point0 == None:
            point0 = (self.width / 2, self.height / 2, self.depth / 2)
        point1 = (width / 2, height / 2, depth / 2)
        self.width = round(width)
        self.height = round(height)
        self.depth = round(depth)
        keys = self._neuron_node_list.keys()
        for key in keys:
            neuron_node = self._neuron_node_list[key]
            x = neuron_node.x - point0[0]
            y = neuron_node.y - point0[1]
            z = neuron_node.z - point0[2]
            neuron_node.x, neuron_node.y, neuron_node.z = rotate_3d((x,y,z), angle_Z, angle_Y, angle_X)
            neuron_node.x = neuron_node.x + point1[0]
            neuron_node.y = neuron_node.y + point1[1]
            neuron_node.z = neuron_node.z + point1[2]
            self._neuron_node_list[key] = neuron_node
        return self

    def resize(self, shape_new = None):
        """
        将神经元节点列表所处的立方体的尺寸缩放为 shape_new
        :param shape_new: depth(Z), height(Y), width(X)
        :return:
        """
        r_z = shape_new[0] / self.depth
        r_y = shape_new[1] / self.height
        r_x = shape_new[2] / self.width
        r_r = min(r_x, r_y, r_z)
        for key in self.keys():
            node = self._neuron_node_list[key]
            node.x = node.x * r_x
            node.y = node.y * r_y
            node.z = node.z * r_z
            node.radius = max(node.radius * r_r, 1) if node.radius != 0 else 0
            self._neuron_node_list[key] = node
        self.depth = shape_new[0]
        self.height = shape_new[1]
        self.width = shape_new[2]
        return self

    def generate_label_3d(self, r_offset = 2, label_mark = 1):
        """
        按照所有节点信息，生成3D标签矩阵；
        这个函数在运行过程中，会破坏self._neuron_node_list中每个神经节点的属性值child_id
        因此在本函数最后重新刷新了self._neuron_node_list中每个神经节点的属性值child_id
        :param r_offset: int，每个节点半径的增加量
        :return:
        """
        image_3d = np.zeros((self.depth, self.height, self.width), dtype=np.uint8)
        keys = list(self._neuron_node_list.keys())
        while keys != []:
            key = keys[0]
            while key:
                neuron_node = self._neuron_node_list[key]
                #print('{} -- ({},{},{}) -- {}/{} - {}'.format(neuron_node.id, neuron_node.x, neuron_node.y, neuron_node.z,
                                                              #neuron_node.child_id, neuron_node.processed, neuron_node.p_id))
                if neuron_node.processed == 1 and neuron_node.child_id == []:
                    #节点本身先于其所有子节点被处理的节点（即有多个子节点），最终在此处被删除
                    #print('{} -- ({},{},{}) -- {}/{} - {} remove 0'.format(neuron_node.id, neuron_node.x, neuron_node.y, neuron_node.z,
                                                              #neuron_node.child_id, neuron_node.processed, neuron_node.p_id))
                    keys.remove(key)
                    break
                elif neuron_node.processed == 1 and neuron_node.child_id != []:
                    #节点本身已被处理，但仍有子节点未被处理的，在此处跳转到其子节点的护理
                    key = self._neuron_node_list[key].child_id.pop(0)
                    continue
                elif neuron_node.processed == 0:
                    #节点本身未被处理的，在此处处理
                    r_reset = r_offset + neuron_node.radius
                    points = neuron_node.get_around_points(r_reset = r_reset, shape = self.shape())
                    for point in points:
                        image_3d[point] = label_mark
                    self._neuron_node_list[key].processed = 1
                    if neuron_node.child_id != []:
                        #节点有子节点的，紧接着跳转过去处理其子节点
                        neuron_node_child = self._neuron_node_list[neuron_node.child_id[0]]
                        self._neuron_node_list[key].child_id = neuron_node.child_id
                        points = neuron_node.get_connect_points(neuron_node_child,
                                                                shape=image_3d.shape,
                                                                r_offset = r_offset)
                        for point in points:
                            image_3d[point] = label_mark
                        key = neuron_node.child_id[0]
                        neuron_node.child_id.pop(0)
                        continue
                    else:
                        #节点本身后于其所有子节点被处理的节点（即没有子节点，这也意味着到达了一条神经纤维的末端），在此处被删除
                        #print('{} -- ({},{},{}) -- {}/{} - {} remove 1'.format(neuron_node.id, neuron_node.x, neuron_node.y, neuron_node.z,
                        #                                      neuron_node.child_id, neuron_node.processed, neuron_node.p_id))
                        keys.remove(key)
                        break
        self.label_3d = Image3D()
        self.label_3d.image_3d = image_3d
        self.label_3d.refresh_shape()
        self.refresh_childID()      #刷新修正节点列表里每个节点的属性值child_id
        return self.label_3d


class NeuronNodeList_SWC(NeuronNodeList):
    """
    从一个swc文件生成一个神经元节点列表 NeuronNodeList
    这个类描述某个 swc 文件中保存的所有神经元节点，保存为字典形式
    在某些swc文件中，给出了神经节点的编号、坐标，和神经节点之间的父子关系，但没有给出每个神经节点的半径（此时所有的半径均被设置为1）
    """
    def __init__(self, file_swc, depth, height, width, resolution = [1,1,1], label = False):
        super(NeuronNodeList_SWC, self).__init__()
        assert file_swc.endswith('swc')
        self.swc_file_name = file_swc
        self.depth = depth
        self.height = height
        self.width = width
        self.resolution = resolution
        self.neuron_node_list()
        if label:
            self.label_3d = self.generate_label_3d()        #对应的标签文件，默认为None，由generate_label_3d生成

    def neuron_node_list(self):
        """
        将 file_swc 文件中的每行信息生成 NeuronNode 类型后保存到 self._neuron_node_list 中，以其 ID 为键值
        :return:
        """
        node_line_list = (line.strip() for line in open(self.swc_file_name, 'r') if line[0] != '#')
        for node_line in node_line_list:
            if node_line == '':
                continue
            node = NeuronNode()
            e = node_line.split(' ')
            node.id = e[0]
            node.type = e[1]
            node.x = float(e[2])
            node.y = self.height - float(e[3]) - 1
            node.z = float(e[4])
            node.radius = float(e[5])
            node.p_id = e[6]
            node.child_id = []              #初始化为空，在生成神经元节点列表过程中被赋值，即被真正初始化
            node.processed = 0              #表示当前节点是否被处理，若被处理则这个值被重置为1；
                                            #即便如此，并不表示该节点已经被处理完，只有当该节点以及该节点的所有子节点都被处理了才表示这个节点被处理完了，
                                            #此时可从NeurongNode_List类型中删除这个节点
            node.resolution = self.resolution
            self.add_neuron_node(node)

    def save(self, saved_file_name = None):
        """
        将某个神经元节点列表保存为 swc 文件的格式
        :param saved_file_name:
        :return:
        """
        assert self.height != None
        if saved_file_name == None:
            saved_file_name, suffix = os.path.splitext(self.swc_file_name)
            saved_file_name += ('_1' + suffix)
        super(NeuronNodeList_SWC, self).save(saved_file_name=saved_file_name)

    def rotate(self, angle_Z = 0, angle_Y = 0, angle_X = 0, point0 = None):
        """
        将当前神经节点列表中的每个节点的坐标进行旋转，当前神经节点列表的数据被改变
        :param angle_Z: 弧度值，绕 Z 轴的旋转角，默认值是最大建议值
        :param angle_Y: 弧度值，向 Y 轴方向的旋转角，默认值是最大建议值
        :param angle_X: 弧度值，向 X 轴方向的旋转角，默认值是最大建议值
        :param point0: 以这个点为旋转点
        :return:
        """
        assert self.width != None and self.depth != None and self.height != None
        if point0 == None:
            point0 = (self.width / 2, self.height / 2, self.depth / 2)
        return super(NeuronNodeList_SWC, self).rotate(angle_Z = angle_Z,
                                                      angle_Y = angle_Y,
                                                      angle_X = angle_X,
                                                      point0 = point0)

    def change_size_info(self, shape = None, coor = None):
        super(NeuronNodeList_SWC, self).change_size_info(shape=shape, coor=coor)
        return self

    def cut_as_shape(self, shape = None, coor = None):
        neuron_node_list_temp, coor = super(NeuronNodeList_SWC, self).cut_as_shape(shape=shape, coor=coor)
        return neuron_node_list_temp, coor
        
    def cut_as_swc(self, id = None, redundancy = (0,0,0)):
        neuron_node_list_temp, coor = super(NeuronNodeList_SWC, self).cut_as_swc(id=id, redundancy=redundancy)
        return neuron_node_list_temp, coor

    def fusion(self, another_nodelist):
        """
        根据节点距离将两个神经元节点列表进行按距离融合，这两个神经元节点列表通常标记了同一个神经元图像
        :param another_nodelist: 与当前神经元节点列表标记相同神经元图像的、另外一个同类型对象
        :return:
        """
        assert isinstance(another_nodelist, NeuronNodeList)
        length = self.__len__()
        for key in another_nodelist._neuron_node_list.keys():
            node = another_nodelist._neuron_node_list[key]
            d0 = math.sqrt(self.width ** 2 + self.height ** 2 + self.depth ** 2)
            node_ = None
            for key_ in self._neuron_node_list:
                node_t = self._neuron_node_list[key_]
                d0_, _, _, _ = node.distance_with_another(node_t)
                if d0_ < d0:
                    d0 = d0_
                    node_ = node_t
            if node_ == None:
                continue
            d0, d1, d2, d3 = node.distance_with_another(node_)
            print('---- {} -- {} -- {} -- {} ----'.format(d0,d1,d2,d3))
            if d2 >= 0.25 or d3 >= 0.25:
                node_.radius = max(node_.radius, node.radius)
                self._neuron_node_list[node_.id] = node_
            else:
                length += 1
                node.p_id = node_.id
                node.id = str(length)
                self.add_neuron_node(node)
        return self


class Image3D():
    """
    这个类型描述一个三维图像，这个三维图像数据的第一个维度描述图像的层数（Z轴，通道数），第二个描述高度（Y轴，行数），第三个描述宽度（X轴，列数）
    """
    def __init__(self):
        self.image_3d = None            # 三维 np.narray 类型
        self.depth, self.height, self.width = None, None, None

    def shape(self):
        return (self.depth, self.height, self.width)

    def set_size(self, shape):
        assert len(shape) == 3
        self.depth, self.height, self.width = shape

    def check_size_info(self):
        """
        检查当前是否有数据，以及数据尺寸是否和相应的属性一致
        :return:
        """
        assert isinstance(self.image_3d, np.ndarray)
        shape = self.image_3d.shape
        assert self.depth == shape[0]
        assert self.height == shape[1]
        assert self.width == shape[2]

    def save(self, image_save_root = None, dim = 0):
        """
        将当前三维图像数据进行保存
        :param image_save_root: string 保存路径
        :param dim: int, 切片维度
        :return:
        """
        self.check_size_info()
        save_image_3d(self.image_3d, image_save_root=image_save_root, dim=dim)

    def refresh_shape(self):
        """
        获取当前三维图像数据的尺寸信息
        :return:
        """
        if isinstance(self.image_3d, np.ndarray):
            self.depth, self.height, self.width = self.image_3d.shape

    def add_slice(self, image_2d):
        """
        将一个二维图像矩阵贴到当前三维图像数据上
        :param image_2d: np.array
        :return:
        """
        if not isinstance(self.image_3d, np.ndarray):
            height, width = image_2d.shape
            self.image_3d = image_2d.reshape((1,height,width))
        else:
            height, width = image_2d.shape
            assert height == self.height and width == self.width
            image_2d = image_2d.reshape((1, height, width))
            self.image_3d = np.concatenate((self.image_3d, image_2d), axis=0)
        self.refresh_shape()

    def add_image3d(self, another_image_3d):
        """
        将另一个三维图像数据接到当前图像数据后
        :param another_image_3d: np.array
        :return:
        """
        if not isinstance(self.image_3d, np.ndarray):
            self.image_3d = another_image_3d
        else:
            _,height,width = another_image_3d.shape
            assert height == self.height and width == self.width
            self.image_3d = np.concatenate((self.image_3d, another_image_3d))
        self.refresh_shape()

    def concatenate(self, another_image3d):
        """
        将当前 NeuronImage3D 和另一个该类型的对象连接起来
        :param another_image3d:
        :return:
        """
        assert isinstance(another_image3d, Image3D)
        if not isinstance(self.image_3d, np.ndarray):
            self.image_3d = another_image3d.image_3d
        else:
            assert self.height == another_image3d.height and self.width == another_image3d.width
            self.image_3d = np.concatenate((self.image_3d, another_image3d.image_3d))
        self.refresh_shape()

    def rotate(self, angle_Z = 0, angle_Y = 0, angle_X = 0, point0=None, filler = 0):
        """
        将当前三维神经元凸显数据进行旋转，绕 Z 轴旋转 angle_Z（弧度），绕 Y 轴方向旋转 angle_Y（弧度），绕 X 轴方向旋转 angle_X（弧度）
        :param angle_Z: 取值范围 [-pi,pi]
        :param angle_Y: 取值范围 []
        :param angle_X: 取值范围 []
        :param point0: 旋转中心点 X(width), Y(height), Z(depth)
        :return:
        """
        self.check_size_info()
        width, height, depth = getshape_rotate_3d(shape = (self.width, self.height, self.depth),
                                                  angle_Z=angle_Z, angle_Y=angle_Y, angle_X=angle_X)
        depth = round(depth)
        height = round(height)
        width = round(width)
        if point0 == None:
            point0 = (round(self.width / 2), round(self.height / 2), round(self.depth / 2))
        point1 = (round(width / 2), round(height / 2), round(depth / 2))
        print(depth, height, width)
        image_3d_temp = np.zeros(shape = (depth, height, width))
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    x0 = x - point1[0]
                    y0 = y - point1[1]
                    z0 = z - point1[2]
                    (xx,yy,zz) = rotate_3d_inverse((x0,y0,z0), angle_Z = angle_Z, angle_Y = angle_Y, angle_X = angle_X)
                    xx = round(xx) + point0[0]
                    yy = round(yy) + point0[1]
                    zz = round(zz) + point0[2]
                    if xx < 0 or xx >= self.width or yy < 0 or yy >= self.height or zz < 0 or zz >= self.depth:
                        continue
                    else:
                        image_3d_temp[z,y,x] = self.image_3d[zz,yy,xx]
        self.image_3d = image_3d_temp
        self.refresh_shape()
        return self

    def translate(self, coor = None):
        """
        实现平移，将图像的原点从默认的 (0,0,0) 平移至给定的 coor，这里移动的是图像的原点，整个图像的移动
        :param coor: X(width), Y(height), Z(depth)
        :return:
        """
        if coor == None:
            raise ValueError
        depth = self.depth + coor[2]
        height = self.height + coor[1]
        width = self.width + coor[0]
        if depth <= 0 or height <= 0 or width <= 0:
            raise ValueError
        image_3d = np.zeros((depth, height, width))
        x0 = coor[0] if coor[0] > 0 else 0
        x1 = 0       if coor[0] > 0 else -coor[0]
        y0 = coor[1] if coor[1] > 0 else 0
        y1 = 0       if coor[1] > 0 else -coor[1]
        z0 = coor[2] if coor[2] > 0 else 0
        z1 = 0       if coor[2] > 0 else -coor[2]
        image_3d[z0:,y0:,x0:] = self.image_3d[z1:,y1:,x1:]
        self.depth = depth
        self.height = height
        self.width = width
        self.image_3d = image_3d
        return self

    def resize(self, shape_new = None):
        """
        将图像矩阵的尺寸缩放为 shape_new
        :param shape_new: depth(Z), height(Y), width(X)
        :return:
        """
        r_z = self.depth / shape_new[0]
        r_y = self.height / shape_new[1]
        r_x = self.width / shape_new[2]
        image_3d = np.zeros(shape_new)
        for z in range(shape_new[0]):
            for y in range(shape_new[1]):
                for x in range(shape_new[2]):
                    image_3d[z,y,x] = self.image_3d[int(z * r_z), int(y * r_y), int(x * r_x)]
            #print('have processed {} slices'.format(z))
        self.image_3d = image_3d
        self.refresh_shape()
        return self

    def cut(self, shape = None, coor = None):
        """
        对当前图像进行裁剪，裁剪原点为 coor，裁剪尺寸为 shape
        :param shape: (depth, height, width)
        :param coor: (x0,y0,z0)
        :return:
        """
        self.check_size_info()
        if shape == None or coor == None:
            raise ValueError
        assert coor[0] >= 0 and coor[1] >= 0 and coor[2] >= 0
        shape = list(shape)
        shape[2] = self.width - coor[0] if coor[0]+shape[2] >= self.width else shape[2]
        shape[1] = self.height - coor[1] if coor[1]+shape[1] >= self.height else shape[1]
        shape[0] = self.depth - coor[2] if coor[2]+shape[0] >= self.depth else shape[0]
        assert shape[0] >= 0 and shape[1] >= 0 and shape[2] >= 0
        image_3D_cut = Image3D()
        image_3D_cut.image_3d = self.image_3d[coor[2]:(coor[2]+shape[0]), coor[1]:(coor[1]+shape[1]), coor[0]:(coor[0]+shape[2])]
        image_3D_cut.refresh_shape()
        return image_3D_cut


class Image3D_PATH(Image3D):
    """
    将文件路径中保存的一张张二维图像序列转换生成三维图像
    """
    def __init__(self, image_path = None):
        """
        :param image_path: string，保存图像数据的路径名，这个路径下保存的图像尺寸必须相同，这个路径下可以有其他类型文件的存在
        """
        assert os.path.isdir(image_path)
        self.image_path = image_path
        super(Image3D_PATH, self).__init__()
        self.suffix = ['.png', '.jepg', '.jpg', '.tiff', '.bmp', '.tif']
        self.generate_image_3d()

    def generate_image_3d(self):
        """
        将路径中保存的图像数据转换成三维图像数据
        :return:
        """
        assert not isinstance(self.image_3d, np.ndarray)
        image_temp = []
        image_name_list = os.listdir(self.image_path)
        image_name_list.sort()
        for image_name in image_name_list:
            _, suffix = os.path.splitext(image_name)
            if not suffix in self.suffix:
                continue
            image_full_name = os.path.join(self.image_path, image_name)
            image_2d = cv2.imread(image_full_name, 0)
            image_temp.append(image_2d)
            if self.height == None:
                self.height, self.width = image_2d.shape
            else:
                assert (self.height, self.width) == image_2d.shape, '图像路径中的图片尺寸不一致'
        self.image_3d = np.array(image_temp)
        self.refresh_shape()

    def save(self, image_save_root = None, dim = 0):
        """
        将当前三维图像数据进行保存
        :param image_save_root: string 保存路径
        :param dim: int 切片维度
        :return:
        """
        if image_save_root == None:
            image_save_root = self.image_path + '_1'
        if not os.path.isdir(image_save_root):
            os.makedirs(image_save_root)
        super(Image3D_PATH, self).save(image_save_root=image_save_root, dim = dim)


class ImageWithLabel():
    """
    联合三维图像和三维标签的类型，对图像数据和标签进行协同操作
    """
    def __init__(self):
        self.image3d = Image3D()
        self.neuronnodelist = NeuronNodeList()
        self.depth = None
        self.height = None
        self.width = None
        self.label_3d = Image3D()

    def shape(self):
        return (self.depth, self.height, self.width)

    def size(self):
        return self.depth * self.height * self.width

    def map_size(self):
        return self.width * self.height

    def check_size_info(self):
        """
        检查图像以及图像标签是否有数据，以及二者尺寸是否一致
        :return:
        """
        if self.image3d.image_3d == None:
            raise ValueError
        if self.neuronnodelist._neuron_node_list.__len__() == 0:
            raise ValueError
        assert self.image3d.depth == self.neuronnodelist.depth
        assert self.image3d.height == self.neuronnodelist.height
        assert self.image3d.width == self.neuronnodelist.width
        assert self.depth == self.image3d.depth
        assert self.height == self.image3d.height
        assert self.width == self.image3d.width

    def refresh_shape(self):
        """
        刷新（更新）数据的长、宽、高等尺寸信息
        :return:
        """
        assert self.image3d.depth == self.neuronnodelist.depth
        assert self.image3d.height == self.neuronnodelist.height
        assert self.image3d.width == self.neuronnodelist.width
        self.depth = self.image3d.depth
        self.height = self.image3d.height
        self.width = self.image3d.width

    def save(self, image_save_root = None, saved_file_name = None, label_save_root = None):
        """
        保存图像和标签数据
        :param image_save_root: 神经元图像保存路径
        :param saved_file_name: swc 文件保存路径
        :param label_save_root: 标签图像保存路径
        :return:
        """
        if image_save_root == None:
            pass
        else:
            self.image3d.save(image_save_root = image_save_root, dim = 0)
        if saved_file_name == None:
            pass
        else:
            self.neuronnodelist.save(saved_file_name = saved_file_name)
        if label_save_root == None:
            pass
        elif label_save_root == image_save_root:
            raise ValueError('神经元图像数据保存路径和神经元标签图像保存路径不能相同，请重置')
        else:
            if not os.path.isdir(label_save_root):
                os.makedirs(label_save_root)
            self.label_3d.save(image_save_root = label_save_root, dim = 0)

    def rotate(self, angle_Z = 0, angle_Y = 0, angle_X = 0, point0 = None, filler = 0):
        """
        协同旋转
        :param angle_Z: 绕 Z 轴的旋转角度（弧度值）
        :param angle_Y: 绕 Y 轴的旋转角度（弧度值）
        :param angle_X: 绕 X 轴的旋转角度（弧度值）
        :param point0: 旋转中心点，原点平移至此
        :return:
        """
        self.image3d = self.image3d.rotate(angle_Z=angle_Z, angle_Y=angle_Y, angle_X=angle_X, point0=point0, filler = filler)
        self.neuronnodelist = self.neuronnodelist.rotate(angle_Z=angle_Z,
                                                         angle_Y=angle_Y,
                                                         angle_X=angle_X,
                                                         point0=point0)
        self.refresh_shape()
        return self

    def translate(self, coor = None):
        """
        实现平移，将图像的原点从默认的 (0,0,0) 平移至给定的 coor，相应的也要对标签进行处理
        :param coor: X(width), Y(height), Z(depth)
        :return:
        """
        self.image3d = self.image3d.translate(coor = coor)
        self.neuronnodelist = self.neuronnodelist.change_size_info(shape=(self.image3d.depth,
                                                                          self.image3d.height,
                                                                          self.image3d.width),
                                                                   coor=tuple(-x for x in coor))
        self.refresh_shape()
        return self

    def resize(self, shape_new = None):
        """
        将图像矩阵以及相应的标签数据所处立方体的尺寸缩放为 shape_new
        :param shape_new: depth(Z), height(Y), width(X)
        :return:
        """
        self.image3d = self.image3d.resize(shape_new = shape_new)
        self.neuronnodelist = self.neuronnodelist.resize(shape_new = shape_new)
        self.refresh_shape()
        return self

    def cut_as_swc(self, id = None):
        """
        根据节点编号对神经元节点进行裁剪，并相应的从图像数据中裁剪出匹配的三维数据
        :param id:
        :return:
        """
        imagelabel_cut = ImageWithLabel()
        imagelabel_cut.neuronnodelist, coor = self.neuronnodelist.cut_as_swc(id=id)
        imagelabel_cut.image3d = self.image3d.cut(shape = (imagelabel_cut.neuronnodelist.depth,
                                                           imagelabel_cut.neuronnodelist.height,
                                                           imagelabel_cut.neuronnodelist.width),
                                                  coor = coor)
        imagelabel_cut.refresh_shape()
        return imagelabel_cut

    def cut_as_shape(self, shape = None, coor = None):
        """
        从带标签的图像数据中进行裁剪，裁剪的起始坐标为 coor，裁剪形状为 shape
        :param shape:
        :param coor:
        :return:
        """
        imagelabel_cut = ImageWithLabel()
        imagelabel_cut.neuronnodelist, coor = self.neuronnodelist.cut_as_shape(shape = shape, coor = coor)
        imagelabel_cut.image3d = self.image3d.cut(shape = (imagelabel_cut.neuronnodelist.depth,
                                                           imagelabel_cut.neuronnodelist.height,
                                                           imagelabel_cut.neuronnodelist.width),
                                                  coor = coor)
        imagelabel_cut.refresh_shape()
        return imagelabel_cut

    def cut_whole(self):
        """
        将神经元节点列表在其所处的立方体内完整裁剪，并将对应的神经元图像数据进行裁剪
        :return:
        """
        self.neuronnodelist, coor = self.neuronnodelist.cut_whole()
        self.image3d = self.image3d.cut(shape = (self.neuronnodelist.depth,
                                                 self.neuronnodelist.height,
                                                 self.neuronnodelist.width),
                                        coor = coor)
        self.refresh_shape()
        return self

    def fusion_in_label_spatial(self, another_imagewithlabel, coor = (0,0,0), flag = True, labeled = '2'):
        """
        将当前带标签的图像与另一个同类型对象的图像进行空域”融合“，具体是将另一个同类型的数据融合到当前类型的 coor 位置
        :param another_imagewithlabel:
        :param coor: X(width),Y(height),Z(depth), 融合的起始坐标
        :param flag: bool, 融合时候是否避开当前图像上已有的标签数据的坐标位置，若为 True，则避开，否则可重叠
        :return:
        """
        assert isinstance(another_imagewithlabel, ImageWithLabel)
        assert another_imagewithlabel.depth + coor[2] <= self.depth
        assert another_imagewithlabel.height + coor[1] <= self.height
        assert another_imagewithlabel.width + coor[0] <= self.width
        depth, height, width = another_imagewithlabel.shape()
        another_imagewithlabel = another_imagewithlabel.translate(coor=coor)
        another_imagewithlabel.label_3d = another_imagewithlabel.neuronnodelist.generate_label_3d()
        for z in range(coor[2], coor[2] + depth):
            for y in range(coor[1], coor[1] + height):
                for x in range(coor[0], coor[0] + width):
                    if another_imagewithlabel.label_3d.image_3d[z,y,x] == 0:
                        continue
                    if self.label_3d.image_3d[z,y,x] != 0:
                        if flag:
                            continue
                        else:
                            self.image3d.image_3d[z, y, x] = another_imagewithlabel.image3d.image_3d[z, y, x]
                            self.label_3d.image_3d[z, y, x] = int(labeled)
                    else:
                        self.image3d.image_3d[z, y, x] = another_imagewithlabel.image3d.image_3d[z,y,x]
                        self.label_3d.image_3d[z, y, x] = int(labeled)
        return self

    def fusion_in_swc_file(self, another_imagewithlabel, coor = (0,0,0), flag = True, labeled = '2', r_offset = 2):
        """
        将当前带标签的图像与另一个同类型对象的图像进行空域”融合“
        :param another_imagewithlabel:
        :param coor: X(width),Y(height),Z(depth), 融合的起始坐标
        :param flag: bool, 融合时候是否避开当前图像上已有的标签数据的坐标位置，若为 True，则避开，否则可重叠
        :return:
        """
        assert isinstance(another_imagewithlabel, ImageWithLabel)
        assert another_imagewithlabel.depth + coor[2] <= self.depth
        assert another_imagewithlabel.height + coor[1] <= self.height
        assert another_imagewithlabel.width + coor[0] <= self.width
        another_imagewithlabel = another_imagewithlabel.translate(coor = coor)
        node_list_temp = NeuronNodeList()
        length = self.neuronnodelist.__len__()
        for key in another_imagewithlabel.neuronnodelist.keys():
            node = another_imagewithlabel.neuronnodelist._neuron_node_list[key]
            r_reset = r_offset + node.radius
            node.id = str(int(node.id) + length)
            if node.p_id != '-1':
                node.p_id = str(int(node.p_id) + length)
            node.child_id = []
            d0, d1, d2, d3 = self.neuronnodelist.distance_with_neuronnode(node)
            if d1 == 0:
                node_list_temp.add_neuron_node(neuron_node=node)
                points = node.get_around_points(r_reset = r_reset, shape=another_imagewithlabel.shape())
                for point in points:
                    self.image3d.image_3d[point] = another_imagewithlabel.image3d.image_3d[point]
            elif flag:
                if d2 >= 0.75 or d3 >= 0.75:
                    continue
                else:
                    node.radius = node.radius * (1 - d3) / 2
                    node_list_temp.add_neuron_node(neuron_node=node)
                    points = node.get_around_points(r_reset = r_reset, shape=another_imagewithlabel.shape())
                    for point in points:
                        self.image3d.image_3d[point] = another_imagewithlabel.image3d.image_3d[point]
            elif flag == False:
                if d2 >= 0.5 or d3 >= 0.5:
                    node_list_temp.add_neuron_node(neuron_node=node)
                    points = node.get_around_points(r_reset = r_reset, shape=another_imagewithlabel.shape())
                    for point in points:
                        self.image3d.image_3d[point] = another_imagewithlabel.image3d.image_3d[point]
                else:
                    node.radius = node.radius * (1 - d3) / 2
                    node_list_temp.add_neuron_node(neuron_node=node)
                    points = node.get_around_points(r_reset = r_reset, shape=another_imagewithlabel.shape())
                    for point in points:
                        self.image3d.image_3d[point] = another_imagewithlabel.image3d.image_3d[point]
        self.neuronnodelist.concatenate(another_nodelist=node_list_temp)
        self.neuronnodelist.refresh_childID()
        return self


class NeuronImageWithLabel(ImageWithLabel):
    """
    神经元图像数据和标签数据的协同处理类型
    """
    def __init__(self, image_path, file_swc, resolution = [1,1,1/0.32], label = False):
        """
        根据给定的图像路径生成 Image3D_PATH 类型，根据给定的 swc 文件生成 NeuronNodeList_SWC 类型
        :param image_path: 保存神经元的二维图像序列的路径
        :param file_swc: swc 文件全名
        """
        assert os.path.isdir(image_path)
        assert os.path.isfile(file_swc)
        super(NeuronImageWithLabel, self).__init__()
        self.image3d = Image3D_PATH(image_path=image_path)
        self.depth = self.image3d.depth
        self.height = self.image3d.height
        self.width = self.image3d.width
        self.neuronnodelist = NeuronNodeList_SWC(file_swc=file_swc,
                                                 height=self.height,
                                                 width=self.width,
                                                 depth=self.depth,
                                                 resolution = resolution)
        self.label_3d = None
        if label:
            self.label_3d = self.neuronnodelist.generate_label_3d()

    def save(self, image_save_root = None, saved_file_name = None, label_save_root = None):
        """
        保存图像和标签数据
        :param image_save_root:
        :param saved_file_name:
        :param lable_save_root:
        :return:
        """
        super(NeuronImageWithLabel, self).save(image_save_root, saved_file_name, label_save_root)

    def rotate(self, angle_Z = 0., angle_Y = 0., angle_X = 0., point0 = None, label = False, filler = 0):
        """
        协同旋转
        :param angle_Z: 绕 Z 轴的旋转角度（弧度值）
        :param angle_Y: 绕 Y 轴的旋转角度（弧度值）
        :param angle_X: 绕 X 轴的旋转角度（弧度值）
        :param point0: 旋转中心点，原点平移至此
        :param label: bool 确定是否刷新三维标签矩阵数据
        :return:
        """
        super(NeuronImageWithLabel, self).rotate(angle_Z=angle_Z,
                                                 angle_Y=angle_Y,
                                                 angle_X=angle_X,
                                                 point0=point0,
                                                 filler = filler)
        if label:
            self.label_3d = self.neuronnodelist.generate_label_3d()
        return self

    def translate(self, coor = None, label = False):
        """
        实现平移，将图像的原点从默认的 (0,0,0) 平移至给定的 coor，相应的也要对标签进行处理
        :param coor: X(width), Y(height), Z(depth)
        :param label: bool 确定是否刷新三维标签矩阵数据
        :return:
        """
        super(NeuronImageWithLabel, self).translate(coor=coor)
        if label:
            self.label_3d = self.neuronnodelist.generate_label_3d()
        return self

    def resize(self, shape_new = (64,256,256), label = False):
        """
        将图像矩阵以及相应的标签数据所处立方体的尺寸缩放为 shape_new
        :param shape_new: depth(Z), height(Y), width(X)
        :param label: bool 确定是否刷新三维标签矩阵数据
        :return:
        """
        super(NeuronImageWithLabel, self).resize(shape_new=shape_new)
        if label:
            self.label_3d = self.neuronnodelist.generate_label_3d()
        return self

    def cut_as_swc(self, id = None):
        """
        根据节点编号对神经元节点进行裁剪，并相应的从图像数据中裁剪出匹配的三维数据
        :param id:
        :return:
        """
        imagelabel_cut = super(NeuronImageWithLabel, self).cut_as_swc(id = id)
        imagelabel_cut.label_3d = imagelabel_cut.neuronnodelist.generate_label_3d()
        return imagelabel_cut

    def cut_as_shape(self, shape = None, coor = None, label = False):
        """
        从带标签的图像数据中进行裁剪，裁剪的起始坐标为 coor，裁剪形状为 shape
        :param shape:
        :param coor:
        :return:
        """
        imagelabel_cut = super(NeuronImageWithLabel, self).cut_as_shape(shape = shape, coor = coor)
        if label:
            imagelabel_cut.label_3d = self.label_3d.cut(shape = shape, coor = coor)
        return imagelabel_cut

    def cut_swc_shape(self, id=None, label = False):
        """
        根据节点编号对神经元节点进行裁剪，并相应的从图像数据中裁剪出匹配的三维数据，且根据相应的形状参数对标签数据也进行裁剪
        :param id:
        :return:
        """
        imagelabel_cut = ImageWithLabel()
        imagelabel_cut.neuronnodelist, coor = self.neuronnodelist.cut_as_swc(id=id)
        imagelabel_cut.neuronnodelist, _ = self.neuronnodelist.cut_as_shape(shape=imagelabel_cut.neuronnodelist.shape(), coor = coor)
        imagelabel_cut.image3d = self.image3d.cut(shape = imagelabel_cut.neuronnodelist.shape(), coor = coor)
        if label:
            imagelabel_cut.label_3d = self.label_3d.cut(shape = imagelabel_cut.neuronnodelist.shape(), coor = coor)
        imagelabel_cut.refresh_shape()
        return imagelabel_cut

    def cut_whole(self, label = False):
        """
        将神经元节点列表在其所处的立方体内完整裁剪，并将对应的神经元图像数据进行裁剪
        :return:
        """
        super(NeuronImageWithLabel, self).cut_whole()
        if label:
            self.label_3d = self.neuronnodelist.generate_label_3d()
        return self

    def fusion_in_label_spatial(self, another_imagewithlabel, coor = (0,0,0), flag = True, labeled = '2'):
        """
        将当前带标签的图像与另一个同类型对象的图像进行空域“融合”，具体是将另一个同类型的数据融合到当前类型的 coor 位置
        这个融合方式无法更新相应的 swc 文件，运行较慢；但相对来说更精细
        :param another_imagewithlabel:
        :param coor: X(width),Y(height),Z(depth), 融合的起始坐标
        :param flag: bool, 融合时候是否避开当前图像上已有的标签数据的坐标位置，若为 True，则避开，否则可重叠
        :return:
        """
        super(NeuronImageWithLabel, self).fusion_in_label_spatial(another_imagewithlabel = another_imagewithlabel,
                                                                  coor = coor, flag = flag, labeled = labeled)
        return self

    def fusion_in_swc_file(self, another_imagewithlabel, coor = (0,0,0), flag = True, labeled = '2', r_offset = 2):
        """
        将当前带标签的图像与另一个同类型对象的图像进行空域“融合”
        这个融合方式能够更新 swc 文件、图像数据，并可以生成新的标签图像数据，速度也较快，但不够精细，不过应该够用了。
        :param another_imagewithlabel:
        :param coor: X(width),Y(height),Z(depth), 融合的起始坐标
        :return:
        """
        super(NeuronImageWithLabel, self).fusion_in_swc_file(another_imagewithlabel=another_imagewithlabel,
                                                             coor = coor, flag = flag,
                                                             labeled = labeled, r_offset = r_offset)
        self.label_3d = self.neuronnodelist.generate_label_3d()
        return self

    def fusion_swc_file(self, another_):
        """
        将当前对象的神经元节点列表与另一神经元节点列表，或是另一个同类型对象的神经元节点列表进行按距离融合
        :param another_: NeuronNodeList_SWC 类型或是 ImageWithLabel类型
                        它与当前对象或是有相同的神经元图像数据，或是标记了相同过的神经元图像数据
        :return:
        """
        if isinstance(another_, NeuronNodeList_SWC):
            self.neuronnodelist.fusion(another_)
            return self
        elif isinstance(another_, ImageWithLabel):
            self.neuronnodelist.fusion(another_.neuronnodelist)
            return self
        else:
            raise TypeError


def generate_label(root = '/home/li-qiufu/PycharmProjects/MyDataBase/0_HUST_DATA/4_image_label'):
    sub_pathes = ['angle_0', 'angle_60', 'angle_120', 'angle_180', 'angle_240', 'angle_300']
    neuron_name_list = ['N041', 'N042', 'N043', 'N044', 'N056', 'N068', 'N075']
    for sub_path in sub_pathes:
        for neuron_name in neuron_name_list:
            image_path = os.path.join(root, sub_path, neuron_name, 'image')
            print('generate_label - processing -- {}'.format(image_path))
            label_path = os.path.join(root, sub_path, neuron_name, 'label')
            file_swc = os.path.join(root, sub_path, neuron_name, neuron_name + '.swc')
            file_swc_noise = os.path.join(root, sub_path, neuron_name, 'noise.swc')
            image_3d = Image3D_PATH(image_path = image_path)
            neuron_list = NeuronNodeList_SWC(file_swc = file_swc,
                                             depth = image_3d.depth,
                                             height = image_3d.height,
                                             width = image_3d.width,
                                             resolution = [1,1,1])
            neuron_list.generate_label_3d(r_offset = 5, label_mark = 1)
            neuron_list_noise = NeuronNodeList_SWC(file_swc = file_swc_noise,
                                                   depth = image_3d.depth,
                                                   height = image_3d.height,
                                                   width = image_3d.width,
                                                   resolution = [1,1,1])
            neuron_list_noise.generate_label_3d(r_offset = 2, label_mark = 2)
            label_3d = neuron_list.label_3d.image_3d + neuron_list_noise.label_3d.image_3d
            label_3d[label_3d == 3] = 1
            save_image_3d(label_3d, label_path)

def resize(root = '/home/li-qiufu/PycharmProjects/MyDataBase/0_HUST_DATA/4_image_label',
           save_root = '/home/li-qiufu/PycharmProjects/MyDataBase/0_HUST_DATA/5_resized',
           size = (32,128,128)):
    sub_pathes = ['angle_0', 'angle_60', 'angle_120', 'angle_180', 'angle_240', 'angle_300']
    neuron_name_list = ['N041', 'N042', 'N043', 'N044', 'N056', 'N068', 'N075']
    for sub_path in sub_pathes:
        for neuron_name in neuron_name_list:
            image_path = os.path.join(root, sub_path, neuron_name, 'image')
            print('resize - processing -- {}'.format(image_path))
            label_path = os.path.join(root, sub_path, neuron_name, 'label')
            image_path_save = os.path.join(save_root, sub_path, neuron_name, 'image')
            label_path_save = os.path.join(save_root, sub_path, neuron_name, 'label')
            image_3d = Image3D_PATH(image_path = image_path)
            label_3d = Image3D_PATH(image_path = label_path)
            image_3d.resize(shape_new = size).save(image_save_root = image_path_save)
            label_3d.resize(shape_new = size).save(image_save_root = label_path_save)



if __name__ == '__main__':
    """
    root = '/media/li-qiufu/Neuron_Data_10T/Neuron_image_3d'
    root_0 = '/home/li-qiufu/PycharmProjects/MyDataBase/0_HUST_DATA/'
    neuron_names = ['N085', 'N086', 'N087', 'N102']
    sub_path_0 = '0_resized'
    sub_path_1 = '1_rotated'
    sub_path_2 = '2_cuted'
    sub_path_3 = '3_cuted'
    neuron_name = 'N075'
    image_save_root = os.path.join(root, neuron_name)
    saved_file_name = os.path.join(root, neuron_name, neuron_name + '.swc')
    image_save_root_0  = os.path.join(root_0, sub_path_0, neuron_name, 'image')
    saved_file_name_0 = os.path.join(root_0, sub_path_0, neuron_name, neuron_name + '.swc')
    image_save_root_1  = os.path.join(root_0, sub_path_1, neuron_name, 'image')
    saved_file_name_1 = os.path.join(root_0, sub_path_1, neuron_name, neuron_name + '.swc')
    image_save_root_2  = os.path.join(root_0, sub_path_2, neuron_name, 'image')
    saved_file_name_2 = os.path.join(root_0, sub_path_2, neuron_name, neuron_name + '.swc')
    image_save_root_3  = os.path.join(root_0, sub_path_3, neuron_name, 'image')
    saved_file_name_3 = os.path.join(root_0, sub_path_3, neuron_name, neuron_name + '.swc')

    NIL = NeuronImageWithLabel(image_path = image_save_root_2, file_swc = saved_file_name_2)
    #NIL = NIL.resize(shape_new = (144, 1000, 1000))
    #NIL = NIL.rotate(angle_Z = math.pi * 64 / 180)
    NIL = NIL.cut_as_shape(coor = (321, 0, 0), shape = (NIL.depth, NIL.height, NIL.width - 321))
    #NIL = NIL.cut_whole()
    NIL.save(image_save_root = image_save_root_3, saved_file_name = saved_file_name_3)
    """
    #image_path = '/home/li-qiufu/Downloads/module_12/original_image'
    #label_path = '/home/li-qiufu/Downloads/module_12/label'
    #file_swc = '/home/li-qiufu/Downloads/module_12/target.swc'
    #NIL = NeuronImageWithLabel(image_path = image_path, file_swc = file_swc, resolution = [1,1,1/0.32], label = True)
    #NIL.save(label_save_root = label_path)
    #generate_label()
    #resize()
    file_swc_0 = '/home/li-qiufu/PycharmProjects/MyDataBase/paper_data/0610/000000/000000.swc'
    label_path = '/home/li-qiufu/PycharmProjects/MyDataBase/paper_data/0610/000000/label'
    #file_swc_1 = '/home/li-qiufu/PycharmProjects/MyDataBase/paper_data/old/Model_4/DataBase_16/angle_0/000003/neuron_1/000004.tiff_import.tif_x429_y321_z50_app2.swc'
    #file_swc_2 = '/home/li-qiufu/PycharmProjects/MyDataBase/paper_data/old/Model_4/DataBase_16/angle_0/000003/neuron_1/000004.swc'
    NNL_0 = NeuronNodeList_SWC(file_swc = file_swc_0, height = 322, width = 465, depth = 155)
    NNL_0.generate_label_3d()
    #NNL_1 = NeuronNodeList_SWC(file_swc = file_swc_1, height = 438, width = 504, depth = 189)
    NNL_0.label_3d.save(image_save_root = label_path)