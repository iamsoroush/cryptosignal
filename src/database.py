from datetime import datetime
import pytz

from peewee import SqliteDatabase, Model, DateTimeField, CharField
from src.utils import get_logger


logger = get_logger(name=__name__, write_logs=True)


class DBHandler:

    db = SqliteDatabase('saita_bot.db')

    def __init__(self):
        # if logger:
        #     self.logger = logger
        # else:
        #     self.logger = get_logger(name=__name__, write_logs=True)
        self.logger = logger

    def start(self):
        # DBHandler.db.connect()
        DBHandler.db.create_tables([User], safe=True)
        # self.logger.info('connected.')

    def add_user(self, user_id, first_name, last_name, username, chat_id):
        query = User.select().where(User.user_id == user_id)
        if query.exists():
            self.logger.warning('user {} already exists.'.format(user_id))
            return query.get()
        else:
            user = User.create(user_id=user_id,
                               first_name=first_name,
                               last_name=last_name,
                               username=username,
                               chat_id=chat_id)
            self.logger.info('user {} added to databse.'.format(user))
            return user

    def update_currency_pair(self, new_base, new_target, user_id):
        if self.exists(user_id):
            pair = '{}{}'.format(new_base.upper(), new_target.upper())
            user = self.get_user(user_id)
            user.pair = pair
            # user.target_currency = new_target
            user.save()
            self.logger.info('currency pair updated for user {}'.format(user_id))
            return user
        else:
            # self.logger.warning('user {} does not exist.'.format(user_id))
            raise Exception('user {} does not exist!'.format(user_id))

    def update_time_frame(self, new_tf, user_id):
        if self.exists(user_id):
            user = self.get_user(user_id)
            user.time_frame = new_tf
            user.save()
            self.logger.info('time frame updated for user {}'.format(user_id))
            return user
        else:
            # self.logger.warning('user {} does not exist.'.format(user_id))
            raise Exception('user {} does not exist!'.format(user_id))

    def get_matched_users(self, pair, time_frame):

        """Returns a list of User objects, who match all the constraints."""

        users = User.select().where((User.pair == pair) &
                                    (User.time_frame == time_frame))
        if users.exists():
            return list(users)
        else:
            return list()

    def get_users(self):

        """Returns all the existing users."""

        return User.select()

    @staticmethod
    def exists(user_id):
        return User.select().where(User.user_id == user_id).exists()

    @staticmethod
    def get_user(user_id):
        return User.get(User.user_id == user_id)


class BaseModel(Model):
    class Meta:
        database = DBHandler.db


def get_default_pair():
    return 'BTCUSDT'


def get_default_time_frame():
    return '30m'


class User(BaseModel):
    user_id = CharField(unique=True, primary_key=True)
    chat_id = CharField()
    username = CharField(unique=True, null=True)
    first_name = CharField()
    last_name = CharField(null=True)

    joined_at = DateTimeField(default=datetime.now(pytz.timezone('Asia/Tehran')))

    pair = CharField(default=get_default_pair)
    time_frame = CharField(default=get_default_time_frame)
