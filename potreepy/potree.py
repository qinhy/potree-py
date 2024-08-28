import os 
from enum import Enum
from typing import Callable
import struct
import json
import numpy as np
from typing import List, Union

class Attribute:
    def __init__(self, name: str = "", description: str = "", size: int = 0, num_elements: int = 0, element_size: int = 0, attribute_type=None):
        self.name = name
        self.description = description
        self.size = size
        self.num_elements = num_elements
        self.element_size = element_size
        self.type = {"int8":np.int8,"int16":np.int16,"int32":np.int32,"int64":np.int64,"uint8":np.uint8,"uint16":np.uint16,
                     "uint32":np.uint32,"uint64":np.uint64,"float":np.float32,"double":np.float64,}.get(attribute_type)
        self.min = np.array([ np.inf,  np.inf,  np.inf])
        self.max = np.array([-np.inf, -np.inf, -np.inf])
        
    def __repr__(self):
        return str(self.__dict__)
                
class Attributes:
    def __init__(self, attributes: List[Attribute] = None):
        if attributes is None:
            attributes = []
        self.list = attributes
        self.bytes = sum([attribute.size for attribute in attributes])
        self.pos_scale = np.array([1.0, 1.0, 1.0])
        self.pos_offset = np.array([0.0, 0.0, 0.0])

    def add(self, attribute: Attribute):
        self.list.append(attribute)
        self.bytes += attribute.size

    def get_offset(self, name: str) -> int:
        offset = 0
        for attribute in self.list:
            if attribute.name == name:
                return offset
            offset += attribute.size
        return -1

    def get(self, name: str) -> Union[Attribute, None]:
        for attribute in self.list:
            if attribute.name == name:
                return attribute
        return None
    
    def __repr__(self):
        return str(self.list)

class PotreePoints:
    def __init__(self,metadata_json):
        self.attributes = self.parse_attributes(metadata_json)
        self.attribute_buffers_map = {}
        self.num_points = 0

    def parse_attributes(self,metadata: dict) -> Attributes:
        attribute_list = []
        js_attributes = metadata["attributes"]

        for js_attribute in js_attributes:
            name = js_attribute["name"]
            description = js_attribute["description"]
            size = js_attribute["size"]
            num_elements = js_attribute["numElements"]
            element_size = js_attribute["elementSize"]
            attr_type = js_attribute["type"]

            js_min = np.asarray(list(map(float,js_attribute["min"])))
            js_max = np.asarray(list(map(float,js_attribute["max"])))

            attribute = Attribute(name=name, size=size, num_elements=num_elements, element_size=element_size, attribute_type=attr_type)

            if len(js_min)<=3 and len(js_max)<=3:
                attribute.min[:len(js_min)] = js_min
                attribute.max[:len(js_max)] = js_max
            attribute_list.append(attribute)

        attributes = Attributes(attribute_list)
        attributes.pos_scale = np.array(metadata["scale"])
        attributes.pos_offset = np.array(metadata["offset"])

        return attributes

    def add_attribute_buffer(self, attribute: Attribute, buffer):
        self.attribute_buffers_map[attribute.name] = buffer

    def remove_attribute(self, attribute_name: str):
        index = -1
        for i, attribute in enumerate(self.attributes.list):
            if attribute.name == attribute_name:
                index = i
                break
        if index >= 0:
            del self.attributes
            del self.attribute_buffers_map[attribute_name]

    def get_raw_data(self,key):
        if key not in self.attribute_buffers_map.keys():
            raise ValueError(f'no {key} data!')
            # return None,None
        buffer = self.attribute_buffers_map[key]
        attr = self.attributes.get(key)
        # print(attr.name,attr.type,attr.num_elements)
        ps = np.frombuffer(buffer.data, dtype=attr.type, count=self.num_points*attr.num_elements)
        return attr,ps.reshape(-1,attr.num_elements)

    def get_position(self) -> np.ndarray:
        attr,position = self.get_raw_data('position')
        position = position * self.attributes.pos_scale + self.attributes.pos_offset
        return position
    
    def get_rgb(self) -> np.ndarray:
        attr,rgb = self.get_raw_data('rgb')
        return rgb /attr.max.max()
    
    def get_point_source_id(self) -> np.ndarray:
        attr,ids = self.get_raw_data('point source id')
        return ids
    
    def get_intensity(self) -> np.ndarray:
        attr,intensity = self.get_raw_data('intensity')
        return intensity /attr.max.max()
        
    def get_classification(self) -> np.ndarray:
        attr,classification = self.get_raw_data('classification')
        return classification
    
    def get_area(self):
        ps = self.get_position()
        return np.prod(ps.max(0)[:2] - ps.min(0)[:2])
    
    def get_volumn(self):
        ps = self.get_position()
        return np.prod(ps.max(0)[:3] - ps.min(0)[:3])    
    
    # try:
    #     from PointCloud import PointCloud       
    #     def get_pointcloud(self, rgb=False,intensity=False,classification=False,normals=False):#,row_column_index=False):     
    #         xyz = self.get_position()
    #         rgb = self.get_rgb() if rgb else None
    #         intensity = self.get_intensity() if intensity else None
    #         classification = self.get_classification() if classification else None
    #         pcd = PointCloud(xyz,rgb=rgb,intensity=intensity,labels=classification)
    #         if normals:pcd.estimate_normals()
    #         return pcd            
    # except Exception as e:
    #     print(e,'no PointCloud module, def get_pointcloud(...) not available')

class PotreeNode:    
    class NodeType(Enum):
        NORMAL = 0
        LEAF = 1
        PROXY = 2

    class AABB:
        def __init__(self, min_coords, max_coords):
            self.min = np.asarray(min_coords)
            self.max = np.asarray(max_coords)
            
        def __str__(self):
            return str((self.min,self.max))
        
        def area(self):
            return np.prod(self.max[:2] - self.min[:2])
        
        def volumn(self):
            return np.prod(self.max - self.min)

    def __init__(self, path='',name = '', aabb = None):
        self.name = name
        self.aabb = aabb
        self.parent = None
        self.children:list[PotreeNode] = [None] * 8
        self.node_type = -1
        self.byte_offset = 0
        self.byte_size = 0
        self.num_points = 0
        self.path = path
        
    def __repr__(self):
        return self.name#str(self.__dict__)

    def level(self):
        return len(self.name) - 1

    def get_all_children(self):
        nodes=[]
        self.traverse(lambda node: nodes.append(node))
        return nodes

    def read_all_children_nodes(self):
        nodes=self.get_all_children()
        pp = [n.read_node() for n in nodes]
        return pp

    def traverse(self, callback: Callable[['PotreeNode'], None]):
        callback(self)
        for child in self.children:
            if child is not None:
                child.traverse(callback)

    def write_node(self,data_dict:dict):#{'classification':np.ones(n0.num_points,dtype=np.uint8)})
        tmpp = self
        while tmpp.path == '':
            tmpp = tmpp.parent
        potree_path = tmpp.path
        with open(os.path.join(potree_path,'metadata.json'),'r') as f:
            attributes_json = json.loads(f.read())
            
        points = PotreePoints(attributes_json)
        points.num_points = self.num_points

        octree_path = os.path.join(potree_path,'octree.bin')
        is_brotli_encoded = (attributes_json["encoding"] == "BROTLI")
        if is_brotli_encoded:
            raise ValueError('brotli encoded is not support!')
        else:
            with open(octree_path, 'r+b') as file:  # note the 'r+b' mode for reading and writing in binary mode            
                attribute_offset = 0
                for attribute in points.attributes.list:
                    attribute_data_size = attribute.size * self.num_points
                    offset_target = 0
                    if attribute.name in data_dict.keys():
                        data:np.ndarray = data_dict[attribute.name]
                        if len(data.flatten().tobytes())!=attribute_data_size:
                            raise ValueError(f'data size:{data.shape} is not in right bytes!(attribute.size:{attribute.size} , num_points:{self.num_points})')
                        for i in range(points.num_points):
                            base_offset = i * points.attributes.bytes + attribute_offset
                            file.seek(self.byte_offset + base_offset)
                            file.write(data[i].tobytes())
                            offset_target += attribute.size
                    attribute_offset += attribute.size
    
    def write_uniform_classification(self, lable=0):
        return self.write_node({'classification':np.ones(self.num_points,dtype=np.uint8)*lable})
                        
    def write_classification(self, data):
        return self.write_node({'classification':data})
    
    def write_uniform_rgb(self, color=(0,0,0)):# 0.0 ~ 1.0
        data = np.ones((self.num_points,3),dtype=np.uint16)
        data[:,0] = int(color[0] * np.iinfo(np.uint16).max)
        data[:,1] = int(color[1] * np.iinfo(np.uint16).max)
        data[:,2] = int(color[2] * np.iinfo(np.uint16).max)
        return self.write_node({'rgb':data})
    
    def write_rgb(self, data):
        return self.write_node({'rgb':data})

    def read_node(self):     
        tmpp = self
        while tmpp.path == '':
            tmpp = tmpp.parent
        potree_path = tmpp.path
        with open(os.path.join(potree_path,'metadata.json'),'r') as f:
            attributes_json = json.loads(f.read())
            
        points = PotreePoints(attributes_json)
        points.num_points = self.num_points

        octree_path = os.path.join(potree_path,'octree.bin')
        with open(octree_path, 'rb') as file:
            file.seek(self.byte_offset)
            data = file.read(self.byte_size)

        is_brotli_encoded = (attributes_json["encoding"] == "BROTLI")
        if is_brotli_encoded:
            raise ValueError('brotli encoded is not support!')
        
            # def dealign24b(mortoncode: int) -> int:
            #     x = mortoncode
            #     x = ((x & 0b001000001000001000001000) >> 2) | ((x & 0b000001000001000001000001) >> 0)
            #     x = ((x & 0b000011000000000011000000) >> 4) | ((x & 0b000000000011000000000011) >> 0)
            #     x = ((x & 0b000000001111000000000000) >> 8) | ((x & 0b000000000000000000001111) >> 0)
            #     x = ((x & 0b000000000000000000000000) >> 16) | ((x & 0b000000000000000011111111) >> 0)
            #     return x

            # decoded_buffer = brotli.decompress(data)
            # offset = 0

            # for attribute in attributes.list:
            #     attribute_data_size = attribute.size * self.num_points
            #     name = attribute.name

            #     buffer = np.empty(attribute_data_size, dtype=np.uint8)

            #     if name == "position":
            #         for i in range(points.num_points):
            #             mc_0, mc_1, mc_2, mc_3 = struct.unpack_from("IIII", decoded_buffer, offset + 16 * i)

            #             X = dealign24b((mc_3 & 0x00FFFFFF) >> 0) \
            #                 | (dealign24b(((mc_3 >> 24) | (mc_2 << 8)) >> 0) << 8)

            #             Y = dealign24b((mc_3 & 0x00FFFFFF) >> 1) \
            #                 | (dealign24b(((mc_3 >> 24) | (mc_2 << 8)) >> 1) << 8)

            #             Z = dealign24b((mc_3 & 0x00FFFFFF) >> 2) \
            #                 | (dealign24b(((mc_3 >> 24) | (mc_2 << 8)) >> 2) << 8)

            #             if mc_1 != 0 or mc_2 != 0:
            #                 X = X | (dealign24b((mc_1 & 0x00FFFFFF) >> 0) << 16) \
            #                     | (dealign24b(((mc_1 >> 24) | (mc_0 << 8)) >> 0) << 24)

            #                 Y = Y | (dealign24b((mc_1 & 0x00FFFFFF) >> 1) << 16) \
            #                     | (dealign24b(((mc_1 >> 24) | (mc_0 << 8)) >> 1) << 24)

            #                 Z = Z | (dealign24b((mc_1 & 0x00FFFFFF) >> 2) << 16) \
            #                     | (dealign24b(((mc_1 >> 24) | (mc_0 << 8)) >> 2) << 24)

            #             X32 = np.int32(X)
            #             Y32 = np.int32(Y)
            #             Z32 = np.int32(Z)

            #             buffer[12 * i: 12 * i + 12] = np.array([X32, Y32, Z32], dtype=np.int32).tobytes()

            #         offset += 16 * self.num_points

            #     elif name == "rgb":
            #         for i in range(points.num_points):
            #             mc_0, mc_1 = struct.unpack_from("II", decoded_buffer, offset + 8 * i)

            #             r = dealign24b((mc_1 & 0x00FFFFFF) >> 0) \
            #                 | (dealign24b(((mc_1 >> 24) | (mc_0 << 8)) >> 0) << 8)

            #             g = dealign24b((mc_1 & 0x00FFFFFF) >> 1) \
            #                 | (dealign24b(((mc_1 >> 24) | (mc_0 << 8)) >> 1) << 8)

            #             b = dealign24b((mc_1 & 0x00FFFFFF) >> 2) \
            #                 | (dealign24b(((mc_1 >> 24) | (mc_0 << 8)) >> 2) << 8)

            #             buffer[6 * i: 6 * i + 6] = np.array([r, g, b], dtype=np.uint16).tobytes()

            #         offset += 8 * self.num_points

            #     else:
            #         buffer = np.frombuffer(decoded_buffer[offset:offset + attribute_data_size], dtype=np.uint8)
            #         offset += attribute_data_size

            #     points.add_attribute_buffer(attribute, buffer)

        else:
            attribute_offset = 0
            for attribute in points.attributes.list:
                attribute_data_size = attribute.size * self.num_points
                buffer = np.empty(attribute_data_size, dtype=np.uint8)
                offset_target = 0
                for i in range(points.num_points):
                    base_offset = i * points.attributes.bytes + attribute_offset
                    raw = data[ base_offset : base_offset + attribute.size]
                    buffer[offset_target:offset_target + attribute.size] = np.frombuffer(raw, dtype=np.uint8)
                    offset_target += attribute.size

                points.add_attribute_buffer(attribute, buffer)
                attribute_offset += attribute.size

        return points

class Potree:
    def __init__(self, path=None):
        print(os.path.join(path,'metadata.json'))
        assert os.path.isfile(os.path.join(path,'metadata.json'))
        assert os.path.isfile(os.path.join(path,'hierarchy.bin'))
        assert os.path.isfile(os.path.join(path,'octree.bin'))

        self.root = None
        self.nodes = []
        self.path = path
        if path is not None:
            self.load()
    
    def load_hierarchy_recursive(self, root: PotreeNode, data: bytes, offset: int, size: int):
        bytesPerNode = 22
        numNodes = size // bytesPerNode

        nodes = [root]

        for i in range(numNodes):
            current = nodes[i]

            offsetNode = offset + i * bytesPerNode
            type, childMask, numPoints, byteOffset, byteSize = struct.unpack_from('<BBIqq', buffer=data, offset=offsetNode)

            current.byte_offset = byteOffset
            current.byte_size = byteSize
            current.num_points = numPoints
            current.node_type = type

            if current.node_type == PotreeNode.NodeType.PROXY.value:
                self.load_hierarchy_recursive(current, data, byteOffset, byteSize)
            else:
                for childIndex in range(8):
                    childExists = ((1 << childIndex) & childMask) != 0

                    if not childExists:
                        continue

                    childName = current.name + str(childIndex)

                    child = PotreeNode(name=childName, aabb=self.child_AABB(current.aabb, childIndex))
                    current.children[childIndex] = child
                    child.parent = current

                    nodes.append(child)

    def child_AABB(self, aabb, index):
        min_coords,max_coords = aabb.min.copy(),aabb.max.copy()
        size = [max_coord - min_coord for max_coord, min_coord in zip(aabb.max, aabb.min)]
        min_coords[2] += ( size[2] / 2 if (index & 0b0001) > 0 else -(size[2] / 2) )
        min_coords[1] += ( size[1] / 2 if (index & 0b0010) > 0 else -(size[1] / 2) )
        min_coords[0] += ( size[0] / 2 if (index & 0b0100) > 0 else -(size[0] / 2) )
        return PotreeNode.AABB(min_coords, max_coords)

    def load(self, path=None):
        if self.path is not None:path = self.path        
        assert self.path is not None or path is not None
            
        with open(os.path.join(path,'metadata.json'),'r') as f:
            metadata = json.loads(f.read())

        with open(os.path.join(path,'hierarchy.bin') ,'rb') as f:
            data = f.read()

        jsHierarchy = metadata["hierarchy"]
        firstChunkSize = jsHierarchy["firstChunkSize"]
        # stepSize = jsHierarchy["stepSize"]
        # depth = jsHierarchy["depth"]

        aabb = PotreeNode.AABB(metadata["boundingBox"]["min"],metadata["boundingBox"]["max"])
        self.root = PotreeNode(path, name="r", aabb=aabb)
        self.load_hierarchy_recursive(self.root, data, offset = 0, size = firstChunkSize)
        self.nodes = []
        self.root.traverse(lambda node: self.nodes.append(node))
        return self

    def bfs(self,node=[],depth=0,resdict={}):
        node = list(filter(lambda x: x is not None, node))
        if len(node)==0:return
        res = []
        resdict[depth] = node
        for i in resdict[depth]:
            res += i.children
        self.bfs(res,depth+1,resdict)
    
    def get_nodes_LOD_dict(self):
        res = {}
        self.bfs([self.root],0,res)
        return res
        
    def get_max_LOD(self):
        return max(self.get_nodes_LOD_dict().keys())
    
    def get_nodes_by_LOD(self, lod=0):
        assert type(lod) == int, 'nodes key must be int!'
        return self.get_nodes_LOD_dict().get(lod,[])
    
    def _potree_read_node(self,x):
        return x.read_node()

    def get_point_size_by_LOD(self,lod=0):
        return sum([n.num_points for n in self.get_nodes_by_LOD(lod)])

    def get_data_by_LOD(self,data_name=['position'],lod=0):        
        # multiproc=False
        res = []
        nodes = self.get_nodes_by_LOD(lod)
        # print('read points :',sum([n.num_points for n in nodes]))         
        if len(nodes)==0:return res
        # with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:       
            # (pool.map(self._potree_read_node, nodes) if multiproc else 
        nodes = [n.read_node() for n in nodes]
        method_list = [func for func in dir(PotreePoints) if callable(getattr(PotreePoints, func))]
        for name in data_name:
            name = name.replace(' ', '_')
            if 'get_'+name in method_list:
                ps = [getattr(n, 'get_'+name)() for n in nodes]
                res.append(np.vstack(ps))    
            else:
                raise ('not function get_'+name)
        return res
    
    def get_position_by_LOD(self, lod=0):
        res = self.get_data_by_LOD(data_name=['position'],lod=lod)
        return res[0]
    
    def get_rgb_by_LOD(self, lod=0):
        res = self.get_data_by_LOD(data_name=['rgb'],lod=lod)
        return res[0]
    
    def get_intensity_by_LOD(self, lod=0):
        res = self.get_data_by_LOD(data_name=['intensity'],lod=lod)
        return res[0]