require 'nn'
require 'cunn'
require 'cudnn'
require 'image'
local optnet = require 'optnet'
local tools = require 'tools/tools'
local cluster = require 'tools/clustering'

-- global variables
local im
local outp, out_segm, out_depth, out_instances, labels_segm, labels_inst
local im_path, input_path, output_path


-- load model
local model = torch.load('./models/fastSceneSegmentationFinal.t7')
model:evaluate()
model:cuda()

-- optimize model for inference
optnet.optimizeMemory(model, torch.CudaTensor(1, 3, 600, 1200), {inplace = true, mode = 'inference', removeGradParams = true})
--optnet.optimizeMemory(model, torch.CudaTensor(1, 3, opts.size / 2, opts.size), {inplace = true, mode = 'inference', removeGradParams = true})


function load_image()
	im = image.load(image_path, 3, 'float')
	local _ = im:size()
	im = im:resize(1, _[1], _[2], _[3]):cuda() -- 1 x 3 x h x w
end


function run()
	-- forward through model
	outp = model:forward(im)

	-- Extract different outputs from network
	out_segm = outp[1]:float()                                            -- 1 x 20 x h x w
	out_instances = outp[2]:float()                                       -- 1 x 8 x h x w
	out_depth = outp[3]:float()                                           -- 1 x 1 x h x w

	-- Segm: calculate labels
	local _
	_, labels_segm = torch.max(out_segm, 2)
	labels_segm = labels_segm:byte()                                      -- 1 x 1 x h x w

	-- Cluster instances
	labels_inst = cluster.cluster(out_instances, labels_segm:eq(15), 1.5) -- 1 x h x w
end


function save_image()
	local img = {}
	img[1] = torch.add(im:float(), tools.to_color(labels_segm, 21))
	img[2] = torch.add(im:float(), 0.5*tools.to_color(labels_inst, 256))
	img[3] = 0.5*tools.to_color(labels_inst, 256)
	img[4] = tools.to_color(labels_segm, 21)

	for i = 1,4 do
		img[i] = image.scale(img[i]:squeeze(), 2048, 1024, 'simple')
		image.save(paths.concat(output_path, string.format('output-%d.png', i)), img[i])
	end

end
--]]

output_path = '/data8T/aucid/output/'
input_path = '/data8T/aucid/guideDogBackend/input_image/image/'


image_path = input_path	.. 'img.jpg'
load_image()
run()

torch.save(output_path .. 'outdepth.dat', out_depth)
torch.save(output_path .. 'labels_inst.dat', labels_inst)
save_image()


--[[
while (true)
do
	local success, cur
	cur = tostring(os.time() - 8)
	image_path = input_path .. cur .. '.jpg'
	local _ = io.open(image_path, 'r')
	if (_)
	then
		_:close()
		print('get' .. cur)
		--load_image()
		os.execute('curl -d "time=' .. cur .. '&data={somejsondata:1}" https://www.guidedog.ml/postbin')
		--run()
		--print(os.time())
	else
		print(cur)
		os.execute('sleep 0.5')
	end
end
]]--
