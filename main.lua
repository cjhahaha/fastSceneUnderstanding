-- time log
print('starting..', os.time())

-- package path
package.cpath = "/data8T/aucid/guideDogBackend/fastSceneUnderstanding/cv/?.so;" .. package.cpath


-- require
require 'nn'
require 'cunn'
require 'cudnn'
require 'torch'
require 'cutorch'
require 'image'
local optnet = require 'optnet'
local tools = require 'tools/tools'
local cluster = require 'tools/clustering'
local liblabel = require 'liblabel'
local rapidjson = require 'rapidjson'


-- const var
local THRESHOLD = 10000
local LABEL = { -- {{{
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
-- setting
local SAVE_OUTPUT = true -- whether save output
local SAVE_OUTPUT_IMAGE = '/data8T/aucid/guideDogBackend/output/image/img.jpg'
local LABEL_CAR = false


-- global variables
-- im
local im
-- tensors
local outp, out_segm, out_depth, out_instances, labels_segm, labels_inst
local labels, cars, depth
-- paths
local im_path, input_path, output_path
-- model
local model
-- point.txt setting
local n
local output_lists
local f


-- load models, init path
function init_all() -- {{{
	-- load model
	model = torch.load('./models/fastSceneSegmentationFinal.t7')
	--model:evaluate()
	model:cuda()

	-- optimize model for inference
	optnet.optimizeMemory(model, torch.CudaTensor(1, 3, 128, 256), {inplace = true, mode = 'inference', removeGradParams = true})

	-- path setting
	output_path = '../output/'
	input_path = '../input/image/'
end -- }}}


-- load single image
function load_image(image_path) -- {{{
	im = image.load(image_path, 3, 'float')
	local _ = im:size()
	im = im:resize(1, _[1], _[2], _[3]):cuda() -- 1 x 3 x h x w

	n = 0
	output_lists = {}
end -- }}}


-- process single image
function run(car_cluster) -- {{{
	-- forward through model
	outp = model:forward(im)
	
	-- Extract different outputs from network
	out_segm = outp[1]:float()                                            -- 1 x 20 x h x w , confidence
	out_instances = outp[2]:float()                                       -- 1 x 8 x h x w  , confidence
	out_depth = outp[3]:float()                                           -- 1 x 1 x h x w  , depth

	-- Segm: calculate labels
	local _
	_, labels_segm = torch.max(out_segm, 2)
	labels_segm = labels_segm:byte()                                      -- 1 x 1 x h x w  , catagory index [0, 20]

	depth = out_depth[1][1]
	labels = labels_segm[1][1]

	-- Cluster instances
	if car_cluster
	then
		labels_inst = cluster.cluster(out_instances, labels_segm:eq(15), 1.5) -- 1 x h x w      , car
		cars = labels_inst[1]
	end
end -- }}}


-- save image for debug use
function save_image() -- {{{
	local img = {}
	img[1] = torch.add(im:float(), tools.to_color(labels_segm, 21))
	img[2] = torch.add(im:float(), 0.5*tools.to_color(labels_inst, 256))
	img[3] = 0.5*tools.to_color(labels_inst, 256)
	img[4] = tools.to_color(labels_segm, 21)

	local _ = im:size()
	for i = 1,4 do
		img[i] = image.scale(img[i]:squeeze(), _[3], _[4], 'simple')
		image.save(paths.concat(output_path .. 'image/', string.format('output-segm-%d.png', i)), img[i])
	end
end -- }}}


-- save tensor for debug use
function save_tensor() -- {{{
	torch.save(output_path .. 'data/outdepth.dat', out_depth[1][1])
	torch.save(output_path .. 'data/out_instances.dat', out_instances)
	torch.save(output_path .. 'data/labels_segm.dat', labels_segm[1][1])
	torch.save(output_path .. 'data/labels_inst.dat', labels_inst[1])
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
		output_lists[n] = _ .. ' ' .. type_ .. '\n'
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
			processs(_, LABEL[i])
		end
	end
end -- }}}


-- split string with separator
function lua_string_split(output_lists, split_char) -- {{{
	local sub_str_tab = {};
	while (true) do
		local pos = string.find(output_lists, split_char);
		if (not pos) then
			sub_str_tab[#sub_str_tab + 1] = output_lists;
			break;
		end
		local sub_str = string.sub(output_lists, 1, pos - 1);
		sub_str_tab[#sub_str_tab + 1] = sub_str;
		output_lists = string.sub(output_lists, pos + 1, #output_lists);
	end
	return sub_str_tab;
end -- }}}


-- write list.txt
function write_list() -- {{{
	f = assert(io.open('list.txt', 'w'))
	f:write(n, "\n")

	if SAVE_OUTPUT
	then
		f:write(SAVE_OUTPUT_IMAGE, "\n")
	end

	for i=1,n - 1
	do
		f:write(output_lists[i])
	end

	f:close()
end -- }}}


-- load points from points.txt
function load_points() -- {{{
	f = assert(io.open('points.txt'))
	json = {}

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
		if json[cur_label] == nil
		then
			json[cur_label] = {}
		end


		table.insert(json[cur_label], {
			x1 = minx,
			x2 = maxx,
			y1 = miny,
			y2 = maxy,
			dis = math.ceil(dis * 100) / 100
		})
	end


	local rj = rapidjson.encode(json)
	print(rj)

end -- }}}



-- init and load
print('\n init and loading...', os.time())
init_all()
load_image(input_path	.. 'img.jpg')


-- running
print('running...', os.time())
run(LABEL_CAR)


-- label
print('label...', os.time())
if LABEL_CAR
then
	label_car()
end
lable_other()

liblabel:l_label()
load_points()
