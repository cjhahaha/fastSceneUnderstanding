package.cpath = "/data8T/aucid/guideDogBackend/fastSceneUnderstanding/cv/?.so;;"
require 'liblabel'

require 'torch'
require 'image'
local cv = require 'cv'

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

local labels = torch.load('./output/labels_inst.dat')[1]
local depth = torch.load('./output/outdepth.dat')[1][1]


local i = 2
local s = depth:clone()
local cur_lable = labels:clone()

cur_lable:apply(function(l) return l == i and 1 or 0 end)
_ = torch.nonzero(cur_lable):size():size()

s:cmul(cur_lable:float())

--[[
for i=2, 20
do
	local s = depth:clone()
	local cur_lable = labels:clone()

	cur_lable:apply(function(l) return l == i and 1 or 0 end)
	_ = torch.nonzero(cur_lable):size():size()

	s:cmul(cur_lable:float())

	if _ ~= 0 then
		image.save(string.format('./output2/output-%s.png',  LABLE[i]), s)
	end

end
]]--
