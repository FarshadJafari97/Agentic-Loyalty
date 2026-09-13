from store.engine import StoreEnv
from store.models import ProductSpec
from store.schedule import SCHEDULE
from store.catalog import catalog


env = StoreEnv(catalog=catalog, schedule=SCHEDULE)