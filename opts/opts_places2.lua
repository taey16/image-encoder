
local dataset_root = '/data3/Places2/'
local checkpoint_path = paths.concat(dataset_root, 'torch_cache');

local dataset_name = 'Places2'
local total_train_samples = 8097967
local nClasses = 365
local forceClasses = 
  {'a-airfield', 'a-airplane_cabin', 'a-airport_terminal', 'a-alcove', 'a-alley', 'a-amphitheater', 'a-amusement_arcade', 'a-amusement_park', 'a-apartment_building-outdoor', 'a-aquarium', 'a-aqueduct', 'a-arcade', 'a-arch', 'a-archaelogical_excavation', 'a-archive', 'a-arena-hockey', 'a-arena-performance', 'a-arena-rodeo', 'a-army_base', 'a-art_gallery', 'a-art_school', 'a-art_studio', 'a-artists_loft', 'a-assembly_line', 'a-athletic_field-outdoor', 'a-atrium-public', 'a-attic', 'a-auditorium', 'a-auto_factory', 'a-auto_showroom', 'b-badlands', 'b-bakery-shop', 'b-balcony-exterior', 'b-balcony-interior', 'b-ball_pit', 'b-ballroom', 'b-bamboo_forest', 'b-bank_vault', 'b-banquet_hall', 'b-bar', 'b-barn', 'b-barndoor', 'b-baseball_field', 'b-basement', 'b-basketball_court-indoor', 'b-bathroom', 'b-bazaar-indoor', 'b-bazaar-outdoor', 'b-beach', 'b-beach_house', 'b-beauty_salon', 'b-bedchamber', 'b-bedroom', 'b-beer_garden', 'b-beer_hall', 'b-berth', 'b-biology_laboratory', 'b-boardwalk', 'b-boat_deck', 'b-boathouse', 'b-bookstore', 'b-booth-indoor', 'b-botanical_garden', 'b-bow_window-indoor', 'b-bowling_alley', 'b-boxing_ring', 'b-bridge', 'b-building_facade', 'b-bullring', 'b-burial_chamber', 'b-bus_interior', 'b-bus_station-indoor', 'b-butchers_shop', 'b-butte', 'c-cabin-outdoor', 'c-cafeteria', 'c-campsite', 'c-campus', 'c-canal-natural', 'c-canal-urban', 'c-candy_store', 'c-canyon', 'c-car_interior', 'c-carrousel', 'c-castle', 'c-catacomb', 'c-cemetery', 'c-chalet', 'c-chemistry_lab', 'c-childs_room', 'c-church-indoor', 'c-church-outdoor', 'c-classroom', 'c-clean_room', 'c-cliff', 'c-closet', 'c-clothing_store', 'c-coast', 'c-cockpit', 'c-coffee_shop', 'c-computer_room', 'c-conference_center', 'c-conference_room', 'c-construction_site', 'c-corn_field', 'c-corral', 'c-corridor', 'c-cottage', 'c-courthouse', 'c-courtyard', 'c-creek', 'c-crevasse', 'c-crosswalk', 'd-dam', 'd-delicatessen', 'd-department_store', 'd-desert-sand', 'd-desert-vegetation', 'd-desert_road', 'd-diner-outdoor', 'd-dining_hall', 'd-dining_room', 'd-discotheque', 'd-doorway-outdoor', 'd-dorm_room', 'd-downtown', 'd-dressing_room', 'd-driveway', 'd-drugstore', 'e-elevator-door', 'e-elevator_lobby', 'e-elevator_shaft', 'e-embassy', 'e-engine_room', 'e-entrance_hall', 'e-escalator-indoor', 'e-excavation', 'f-fabric_store', 'f-farm', 'f-fastfood_restaurant', 'f-field-cultivated', 'f-field-wild', 'f-field_road', 'f-fire_escape', 'f-fire_station', 'f-fishpond', 'f-flea_market-indoor', 'f-florist_shop-indoor', 'f-food_court', 'f-football_field', 'f-forest-broadleaf', 'f-forest_path', 'f-forest_road', 'f-formal_garden', 'f-fountain', 'g-galley', 'g-garage-indoor', 'g-garage-outdoor', 'g-gas_station', 'g-gazebo-exterior', 'g-general_store-indoor', 'g-general_store-outdoor', 'g-gift_shop', 'g-glacier', 'g-golf_course', 'g-greenhouse-indoor', 'g-greenhouse-outdoor', 'g-grotto', 'g-gymnasium-indoor', 'h-hangar-indoor', 'h-hangar-outdoor', 'h-harbor', 'h-hardware_store', 'h-hayfield', 'h-heliport', 'h-highway', 'h-home_office', 'h-home_theater', 'h-hospital', 'h-hospital_room', 'h-hot_spring', 'h-hotel-outdoor', 'h-hotel_room', 'h-house', 'h-hunting_lodge-outdoor', 'i-ice_cream_parlor', 'i-ice_floe', 'i-ice_shelf', 'i-ice_skating_rink-indoor', 'i-ice_skating_rink-outdoor', 'i-iceberg', 'i-igloo', 'i-industrial_area', 'i-inn-outdoor', 'i-islet', 'j-jacuzzi-indoor', 'j-jail_cell', 'j-japanese_garden', 'j-jewelry_shop', 'j-junkyard', 'k-kasbah', 'k-kennel-outdoor', 'k-kindergarden_classroom', 'k-kitchen', 'l-lagoon', 'l-lake-natural', 'l-landfill', 'l-landing_deck', 'l-laundromat', 'l-lawn', 'l-lecture_room', 'l-legislative_chamber', 'l-library-indoor', 'l-library-outdoor', 'l-lighthouse', 'l-living_room', 'l-loading_dock', 'l-lobby', 'l-lock_chamber', 'l-locker_room', 'm-mansion', 'm-manufactured_home', 'm-market-indoor', 'm-market-outdoor', 'm-marsh', 'm-martial_arts_gym', 'm-mausoleum', 'm-medina', 'm-mezzanine', 'm-moat-water', 'm-mosque-outdoor', 'm-motel', 'm-mountain', 'm-mountain_path', 'm-mountain_snowy', 'm-movie_theater-indoor', 'm-museum-indoor', 'm-museum-outdoor', 'm-music_studio', 'n-natural_history_museum', 'n-nursery', 'n-nursing_home', 'o-oast_house', 'o-ocean', 'o-office', 'o-office_building', 'o-office_cubicles', 'o-oilrig', 'o-operating_room', 'o-orchard', 'o-orchestra_pit', 'p-pagoda', 'p-palace', 'p-pantry', 'p-park', 'p-parking_garage-indoor', 'p-parking_garage-outdoor', 'p-parking_lot', 'p-pasture', 'p-patio', 'p-pavilion', 'p-pet_shop', 'p-pharmacy', 'p-phone_booth', 'p-physics_laboratory', 'p-picnic_area', 'p-pier', 'p-pizzeria', 'p-playground', 'p-playroom', 'p-plaza', 'p-pond', 'p-porch', 'p-promenade', 'p-pub-indoor', 'r-racecourse', 'r-raceway', 'r-raft', 'r-railroad_track', 'r-rainforest', 'r-reception', 'r-recreation_room', 'r-repair_shop', 'r-residential_neighborhood', 'r-restaurant', 'r-restaurant_kitchen', 'r-restaurant_patio', 'r-rice_paddy', 'r-river', 'r-rock_arch', 'r-roof_garden', 'r-rope_bridge', 'r-ruin', 'r-runway', 's-sandbox', 's-sauna', 's-schoolhouse', 's-science_museum', 's-server_room', 's-shed', 's-shoe_shop', 's-shopfront', 's-shopping_mall-indoor', 's-shower', 's-ski_resort', 's-ski_slope', 's-sky', 's-skyscraper', 's-slum', 's-snowfield', 's-soccer_field', 's-stable', 's-stadium-baseball', 's-stadium-football', 's-stadium-soccer', 's-stage-indoor', 's-stage-outdoor', 's-staircase', 's-storage_room', 's-street', 's-subway_station-platform', 's-supermarket', 's-sushi_bar', 's-swamp', 's-swimming_hole', 's-swimming_pool-indoor', 's-swimming_pool-outdoor', 's-synagogue-outdoor', 't-television_room', 't-television_studio', 't-temple-asia', 't-throne_room', 't-ticket_booth', 't-topiary_garden', 't-tower', 't-toyshop', 't-train_interior', 't-train_station-platform', 't-tree_farm', 't-tree_house', 't-trench', 't-tundra', 'u-underwater-ocean_deep', 'u-utility_room', 'v-valley', 'v-vegetable_garden', 'v-veterinarians_office', 'v-viaduct', 'v-village', 'v-vineyard', 'v-volcano', 'v-volleyball_court-outdoor', 'w-waiting_room', 'w-water_park', 'w-water_tower', 'w-waterfall', 'w-watering_hole', 'w-wave', 'w-wet_bar', 'w-wheat_field', 'w-wind_farm', 'w-windmill', 'y-yard', 'y-youth_hostel', 'z-zen_garden'}
local network = 'resception'
local loadSize  = {3, 342, 342}
local sampleSize= {3, 299, 299}
local nGPU = {1,2}
local current_epoch = 1
local current_iter = 0
local test_initialization = false
local retrain_path = 
  false
if retrain_path then
  initial_model = 
    paths.concat(retrain_path, ('model_%d.t7'):format(current_epoch-1)) 
  initial_optimState = 
    paths.concat(retrain_path, ('optimState_%d.t7'):format(current_epoch-1))
else
  initial_model = false
  initial_optimState = false
end

local batchsize = 32
local test_batchsize = 32
local solver = 'nag'
local num_max_epoch = 500
local learning_rate = 0.045
local weight_decay = 0.00002
local learning_rate_decay_seed = 0.94
local learning_rate_decay_start = -1
local learning_rate_decay_every = 40037 * 2
local experiment_id = string.format(
  '%s_X_gpu%d_cudnn-v5_%s_epoch%d_%s_lr%.5f_decay_seed%.3f_start%d_every%d', dataset_name, #nGPU, network, current_epoch, solver, learning_rate, learning_rate_decay_seed, learning_rate_decay_start, learning_rate_decay_every
)

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an vision encoder model')
cmd:text()
cmd:text('Options')

-- dataset specific
cmd:option('-cache', checkpoint_path, 'subdirectory in which to save/log experiments')
cmd:option('-data', dataset_root, 'root of dataset')
cmd:option('-nClasses', nClasses, '# of classes')

-- training specific
cmd:option('-nEpochs', num_max_epoch, 'Number of total epochs to run')
cmd:option('-epochSize', math.ceil(total_train_samples/batchsize), 'Number of batches per epoch')
cmd:option('-epochNumber', current_epoch,'Manual epoch number (useful on restarts)')
cmd:option('-iter_batch', current_iter,'Manual iter number (useful on restarts and lr anneal)')
cmd:option('-batchSize', batchsize, 'mini-batch size (1 = pure stochastic)')
cmd:option('-test_batchSize', test_batchsize, 'test mini-batch size')
cmd:option('-test_initialization', test_initialization, 'test_initialization')
cmd:option('-retrain', initial_model, 'provide path to model to retrain with')
cmd:option('-optimState', initial_optimState, 'provide path to an optimState to reload from')

-- optimizer specific
cmd:option('-solver', solver, 'nag | adam | sgd')
cmd:option('-LR', learning_rate, 
  'learning rate; if set, overrides default LR/WD recipe')
cmd:option('-learning_rate_decay_seed', learning_rate_decay_seed,
  'decay_factor = math.pow(opt.learning_rate_decay_seed, frac)')
cmd:option('-learning_rate_decay_start', learning_rate_decay_start,
  'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', learning_rate_decay_every,
  'every how many iterations thereafter to drop LR by half?')
cmd:option('-momentum', 0.9,  'momentum')
cmd:option('-weightDecay', weight_decay, 'weight decay')

-- network specific
cmd:option('-netType', network, 'Options: [inception_v3 | resception]')

-- misc.
cmd:option('-nDonkeys', 4, 'number of donkeys to initialize (data loading threads)')
cmd:option('-manualSeed', 999, 'Manually set RNG seed')
cmd:option('-display', 5, 'interval for printing train loss per minibatch')
cmd:option('-snapshot', 25000, 'interval for conditional_save')
cmd:text()

local opt = cmd:parse(arg)
opt.loadSize = loadSize
opt.sampleSize= sampleSize
opt.nGPU = nGPU

if not os.execute('cd ' .. opt.data) then
  error(("could not chdir to '%s'"):format(opt.data))
end

opt.save = paths.concat(opt.cache, experiment_id)
os.execute('mkdir -p ' ..opt.save)
print('===> checkpoint path: ' .. opt.save)

return opt
