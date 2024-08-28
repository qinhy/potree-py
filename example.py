from potreepy import Potree
pt = Potree('./data/coordinate')
position,rgb = pt.get_data_by_LOD(['position','rgb'],0)
print(position)
print(rgb)
# you can also try open3d to show results
