package.cpath = "/data8T/aucid/guideDogBackend/fastSceneUnderstanding/cv/?.so;" .. package.cpath

require 'cutorch'
require 'torch'
require 'image'
local liblabel = require 'liblabel'

local THRESHOLD = 10000
local LABLE = {
	'Unlabeled',
	'Road',
	'Sidewalk',
	'Building',
	'Wall',
	'Fence',
	'Pole',
	'TrafficLight',
	'TrafficSign',
	'Vegetation',
	'Terrain',
	'Sky',
	'Person',
	'Rider',
	'Car',
	'Truck',
	'Bus',
	'Train',
	'Motorcycle',
	'Bicycle'
}

local f


print('loading..')
local labels = torch.load('../output/data/labels_segm.dat')
local cars = torch.load('../output/data/labels_inst.dat')
local depth = torch.load('../output/data/outdepth.dat')

local output_path = '/data8T/aucid/guideDogBackend/output/'
local ori_path = '/data8T/aucid/guideDogBackend/output/image/img.jpg'
local out_im_path = '/data8T/aucid/guideDogBackend/output/image/img.jpg'
print('loaded!')


local n = 0
local str = {}


function processs(cur_lable, type_)
	local s = depth:clone()
	local _ = torch.nonzero(cur_lable):size()
	s:cmul(cur_lable:float())

	if _:size() ~= 0 and _[1] > THRESHOLD
	then
		_ = output_path .. string.format('image/output-%s.png', type_)
		image.save(_, s)
		str[n] = _ .. ' ' .. type_ .. '\n'
		n = n + 1
	end
end


function label_car()
	for i=1,cars:max()
	do
		local _ = cars:clone():apply(function(x) return x == i and 1 or 0 end)
		processs(_, string.format("cars-%s", i))
	end
end


function lable_other()
	for i=1,20
	do
		if i ~= 15 then
			local _ = labels:clone():apply(function(x) return x == i and 1 or 0 end)
			processs(_, LABLE[i])
		end
	end
end


function lua_string_split(str, split_char)
	local sub_str_tab = {};
	while (true) do
		local pos = string.find(str, split_char);
		if (not pos) then
			sub_str_tab[#sub_str_tab + 1] = str;
			break;
		end
		local sub_str = string.sub(str, 1, pos - 1);
		sub_str_tab[#sub_str_tab + 1] = sub_str;
		str = string.sub(str, pos + 1, #str);
	end
	return sub_str_tab;
end


function label_all()
	f = assert(io.open('list.txt', 'w'))

	label_car()
	lable_other()

	f:write(n, "\n")
	f:write(ori_path, "\n")
	for i=1,n - 1
	do
		f:write(str[i])
	end

	f:close()

	liblabel:l_label()

	f = assert(io.open('points.txt'))

	local cur_label, minx, miny, maxx, maxy, s, dis
	for l in f:lines()
	do
		l = lua_string_split(l, ' ')
		cur_label = l[1]
		minx = tonumber(l[2] + 1)
		maxx = tonumber(l[3] + 1)
		miny = tonumber(l[4] + 1)
		maxy = tonumber(l[5] + 1)
		s = (maxx - minx + 1) * (maxy - miny + 1)

		dis = depth:sub(miny, maxy, minx, maxx):sum() / s
		print(cur_label, dis)
	end


	print(depth:max(), depth:min())
end


label_all()

