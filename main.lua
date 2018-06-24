require 'nn'
require 'cunn'
require 'cudnn'
require 'image'
local optnet = require 'optnet'
local tools = require 'tools/tools'
local cluster = require 'tools/clustering'

-- load model
local model = torch.load('./models/fastSceneSegmentationFinal.t7')
model:evaluate()
model:cuda()

-- optimize model for inference
--optnet.optimizeMemory(model, torch.CudaTensor(1, 3, opts.size / 2, opts.size), {inplace = true, mode = 'inference', removeGradParams = true})


-- global variables
local im
local outp, out_segm, out_depth, out_instances, labels_segm, labels_inst
local image_path, output_path


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

image_path = '/data8T/aucid/test.jpg'
output_path = '/data8T/aucid/output/'

load_image()
run()
--save_image()
