require 'torch'
require 'image'

local THRESHOLD = 1000
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

local labels = torch.load('/data8T/aucid/guideDogBackend/input_image/output/data/labels_segm.dat')
local cars = torch.load('/data8T/aucid/guideDogBackend/input_image/output/data/labels_inst.dat')
local depth = torch.load('/data8T/aucid/guideDogBackend/input_image/output/data/outdepth.dat')


function label(cur_lable)
	local s = depth:clone()
	local _ = torch.nonzero(cur_lable):size()
	s:cmul(cur_lable:float())

	if _:size() ~= 0 and _[1] > THRESHOLD then
		image.save(string.format('./output2/output-%s.png',  LABLE[i]), s)
		os.execute('./cv/label /data8T/aucid/guideDogBackend/fastSceneUnderstanding/output2/output-' .. LABLE[i] .. '.png ' .. LABLE[i] .. ' /data8T/aucid/guideDogBackend/input_image/image/img1.jpg')
	end
end


function label_car()

end



