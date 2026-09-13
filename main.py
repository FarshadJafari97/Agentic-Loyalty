from store.engine import StoreEnv
from store.models import ProductSpec
from store.schedule import SCHEDULE
from store.catalog import catalog


env = StoreEnv(catalog=catalog, schedule=SCHEDULE)

while not env.finished:
    env.begin_round()

    round_ = env._round

    print(round_)
    # ساخت prompt با products + history
    # اجرای گراف LangGraph
    env.close_round()