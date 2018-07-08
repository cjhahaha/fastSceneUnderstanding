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
--
--[[
local labels = torch.load('./output/out_instances.dat')[1]
local depth = torch.load('./output/outdepth.dat')[1][1]


for i=13, 20
do
	local s = depth:clone()
	local cur_lable = labels[i - 12]

	--cur_lable:apply(function(l) return l == i and 1 or 0 end)
	_ = torch.nonzero(cur_lable):size()

	s:cmul(cur_lable:float())

	if _:size() ~= 0 then --and _[1] > 8000 then
		image.save(string.format('./output2/output-%s.png',  LABLE[i]), s)
		os.execute('./cv/label /data8T/aucid/guideDogBackend/fastSceneUnderstanding/output2/output-' .. LABLE[i] .. '.png ' .. LABLE[i] .. ' /data8T/aucid/guideDogBackend/input_image/image/img1.jpg')
	end
end
-]]
