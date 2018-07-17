-- package path
package.cpath = "/data8T/aucid/guideDogBackend/fastSceneUnderstanding/cv/?.so;" .. package.cpath


-- require
require 'cutorch'
require 'torch'
require 'image'
local liblabel = require 'liblabel'
local rapidjson = require 'rapidjson'


-- const var
local THRESHOLD = 10000
local LABLE = { -- {{{
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
} -- }}}


-- global var
-- tensors
local lables -- labels tensor
local cars   -- cars tensor
local depth  -- depth tensor

-- path setting
local output_path = '/data8T/aucid/guideDogBackend/output/'
local ori_path = '/data8T/aucid/guideDogBackend/output/image/img.jpg'

-- output file setting
local n = 0    -- output n
local str = {} -- output str
local f        -- ouput file contains image path


-- load data
function load_data() -- {{{
	print('loading..')

	labels = torch.load('../output/data/labels_segm.dat')
	cars = torch.load('../output/data/labels_inst.dat')
	depth = torch.load('../output/data/outdepth.dat')

	print('loaded!')
end -- }}}


-- process each label and write image for label.cpp to use
function processs(cur_lable, type_) -- {{{
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
end -- }}}


-- label all cars
function label_car() -- {{{
	for i=1,cars:max()
	do
		local _ = cars:clone():apply(function(x) return x == i and 1 or 0 end)
		processs(_, string.format("cars-%s", i))
	end
end -- }}}


-- label other objects
function lable_other() -- {{{
	for i=1,20
	do
		if i ~= 15 then
			local _ = labels:clone():apply(function(x) return x == i and 1 or 0 end)
			processs(_, LABLE[i])
		end
	end
end -- }}}


-- split string with separator
function lua_string_split(str, split_char) -- {{{
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
end -- }}}


-- write list.txt
function write_list() -- {{{
	f = assert(io.open('list.txt', 'w'))
	f:write(n, "\n")
	f:write(ori_path, "\n")
	for i=1,n - 1
	do
		f:write(str[i])
	end

	f:close()
end -- }}}


-- load points from points.txt
function load_points() -- {{{
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
end -- }}}


function label_all()
	load_data()

	label_car()
	lable_other()

	write_list()

	liblabel:l_label()

	load_points()
end


label_all()

