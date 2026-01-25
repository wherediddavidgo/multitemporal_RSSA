import ee
import time
from tqdm import tqdm

ee.Authenticate()
ee.Initialize(project='ee-wherediddavidgo')

import rssa_utils as utils

# batch parameters, no more than 5 active tasks. If more than 5, wait 60 seconds before uploading.
MAX_IN_FLIGHT = 5
POLL_SECONDS = 60

# Imort assets and imagecollection
buffers = ee.FeatureCollection('projects/ee-wherediddavidgo/assets/ms_grwl_2e4_point_buffers')
rCl = ee.ImageCollection('projects/ee-wherediddavidgo/assets/grwl_centerline_raster')
vCl = ee.FeatureCollection('projects/ee-wherediddavidgo/assets/ms_grwl_cross_sections')
pts = ee.FeatureCollection('projects/ee-wherediddavidgo/assets/ms_grwl_pts_2e4')

big_ic = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 20)


def process_scene_aoi(scene, aoi):
    """Wrapper function trims raster centerline to image bounds, generates masks, calculates widths."""
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
    """Filters geometries to image bounds, gets width of river mask, snow mask, and cloud mask between cross section ends stored in point assets."""
    buffers_filtered = buffers.filterBounds(scene.geometry())
    aoi = buffers_filtered.geometry().intersection(scene.geometry()).dissolve()

    output = ee.Algorithms.If(
        buffers_filtered.size().gt(0),
        process_scene_aoi(scene, aoi),
        None
    )
    return output


def check_in_flight(prefix=None):
    """Returns a list of all tasks queued or running in Google servers."""
    tasks = ee.batch.Task.list()
    in_flight = []
    for t in tasks:
        st = t.status()
        state = st.get('state')
        desc = st.get('description', '')
        if state in ('READY', 'RUNNING'):
            if prefix is None or desc.startswith(prefix):
                # id = desc.split('_')[1]
                # year = desc.split('_')[2]
                in_flight.append(desc)
    return in_flight


def wait_for_room(prefix=None):
    """If too many tasks running or queued, just wait before doing anything else."""
    while len(check_in_flight(prefix)) >= MAX_IN_FLIGHT:
        print('Waiting', POLL_SECONDS)
        time.sleep(POLL_SECONDS)


def check_completed(prefix=None):
    """Looks for completed width extractions. This is necessary so that if the script stops, it can begin where it left off."""
    print("Checking completed tasks")
    tasks = ee.batch.Task.list()
    completed = []
    for t in tqdm(tasks):
        st = t.status()
        state = st.get('state')
        desc = st.get('description', '')
        if state == 'COMPLETED':
            if desc.startswith(prefix):
                # id = desc.split('_')[1]
                # year = desc.split('_')[2]
                completed.append(desc)
    print(f'{len(completed)} / {7 * 344} uploads complete')
    return completed

#  List of mgrs tiles intersecting with MS basin.
f = open('ms_s2_mgrs_ids.txt', 'r')
content = f.read()
idlist = content.split('\n')


completed_tasks = check_completed('widths')
# print(completed_tasks)
active_tasks = check_in_flight('widths')
# print(active_tasks)

props = ['system:index', 'img_id', 'xsec_lengt', 'any', 'cloud_mask', 'cloudwater_mask', 'count', 'endsInWater', 'endsOverEdge', 'river_mask', 'scene_cloudy_pixel_percentage', 'scene_date', 'snow_mask', 'width', 'x', 'y']

# Counter. Checking for active tasks takes a lot of time, so only do it every 10 loops.
n = 0

for year in [2018, 2019, 2020, 2021, 2022, 2023, 2024]:
    for id in idlist:
        task_id = f'widths_{id}_{year}'
        if task_id not in completed_tasks and task_id not in active_tasks:
            if n % 10 == 0:
                wait_for_room('widths')

            # Year nd tile are confirmed not completed or submitted. Proceed with filtering imagecollection and extracting widths.
            temp_ic = big_ic\
                .filter(ee.Filter.calendarRange(year, None, 'year'))\
                .filterMetadata('MGRS_TILE', 'equals', id)
            
            widths = temp_ic.map(EXTRACT_WIDTHS_FROM_IMAGE, dropNulls=True).flatten().select(props)
                
            task = ee.batch.Export.table.toDrive(**{
                'collection': widths,
                'folder': 'ms_rwc_2e4_exports',
                'fileNamePrefix': task_id,
                'description': task_id,

            })

            task.start()
            
            n += 1
            print(f'Year: {year}, Tile: {id} uploaded')

            # somebody online suggested that maybe waiting helps stop GEE from exporting everything to separate folders with the same name.
            time.sleep(1)