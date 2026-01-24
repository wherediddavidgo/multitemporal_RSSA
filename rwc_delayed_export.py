import ee
import time

ee.Authenticate()
ee.Initialize(project='ee-wherediddavidgo')

import rssa_utils as utils

MAX_IN_FLIGHT = 2
POLL_SECONDS = 5

buffers = ee.FeatureCollection('projects/ee-wherediddavidgo/assets/ms_grwl_2e4_point_buffers')
rCl = ee.ImageCollection('projects/ee-wherediddavidgo/assets/grwl_centerline_raster')
vCl = ee.FeatureCollection('projects/ee-wherediddavidgo/assets/ms_grwl_cross_sections')
pts = ee.FeatureCollection('projects/ee-wherediddavidgo/assets/ms_grwl_pts_2e4')

big_ic = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 20)


def process_scene_aoi(scene, aoi):
    rCl_filt = rCl.filterBounds(aoi)\
        .mosaic()\
        .clip(scene.geometry())
    
    scene = utils.ADD_WATER_MASK(scene, aoi, dynamic=True)
    scene = utils.EXTRACT_RIVER(scene, rCl_filt, 1000, 333)
    scene = utils.UNPACK_SCL(scene)
    widths = utils.CALCULATE_WIDTH(scene, pts.filterBounds(scene.geometry()))

    cpp = scene.get('CLOUDY_PIXEL_PERCENTAGE')
    dt = scene.date()

    
    def add_info(f):
        return f.set({'scene_cloudy_pixel_percentage': cpp,
                      'scene_date': dt})
    
    widths = widths.map(add_info)
    
    return widths


def EXTRACT_WIDTHS_FROM_IMAGE(scene):
    buffers_filtered = buffers.filterBounds(scene.geometry())
    aoi = buffers_filtered.geometry().intersection(scene.geometry()).dissolve()

    output = ee.Algorithms.If(
        buffers_filtered.size().gt(0),
        process_scene_aoi(scene, aoi),
        None
    )
    return output


def count_in_flight_tasks(prefix=None):
    tasks = ee.batch.Task.list()
    in_flight = []
    for t in tasks:
        st = t.status()
        state = st.get('state')
        desc = st.get('description', '')
        if state in ('READY', 'RUNNING'):
            if prefix is None or desc.startswith(prefix):
                in_flight.append(t)
    return len(in_flight)


def wait_for_room(prefix=None):
    while count_in_flight_tasks(prefix) >= MAX_IN_FLIGHT:
        print('waiting', POLL_SECONDS)
        time.sleep(POLL_SECONDS)


f = open('ms_s2_mgrs_ids.txt', 'r')
content = f.read()
idlist = content.split('\n')

props = ['system:index', 'img_id', 'xsec_lengt', 'any', 'cloud_mask', 'cloudwater_mask', 'count', 'endsInWater', 'endsOverEdge', 'river_mask', 'scene_cloudy_pixel_percentage', 'scene_date', 'snow_mask', 'width', 'x', 'y']

n = 0

for year in [2018, 2019, 2020, 2021, 2022, 2023, 2024]:
    for id in idlist:
        if n % 3 == 0:
            wait_for_room('widths')


        temp_ic = big_ic\
            .filter(ee.Filter.calendarRange(year, None, 'year'))\
            .filterMetadata('MGRS_TILE', 'equals', id)
        
        widths = temp_ic.map(EXTRACT_WIDTHS_FROM_IMAGE, dropNulls=True).flatten().select(props)
            
        task = ee.batch.Export.table.toDrive(**{
            'collection': widths,
            'folder': 'ms_rwc_2e4_exports',
            'fileNamePrefix': f'widths_{id}_{year}',
            'description': f'widths_{id}_{year}',

        })

        task.start()
        
        n += 1
        print(f'year: {year}, tile: {id} uploaded')