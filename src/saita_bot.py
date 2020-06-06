import os

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ParseMode
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, PicklePersistence, CallbackQueryHandler,\
    ConversationHandler
from telegram.error import BadRequest

from src.utils import get_logger
from src import TIME_FRAMES, BASE_CURRENCY_LIST, TARGET_CURRENCY_LIST


logger = get_logger(name=__name__, write_logs=True)

CHOOSE_PAIR_TF = '1'
CHOOSE_BASE_CURRENCY_STATE = '2'
CHOOSE_TARGET_CURRENCY_STATE = '22'
CHOOSE_TIME_FRAME_STATE = '3'


class SAITABot:

    def __init__(self, bot_token, db):
        global db_handler
        db_handler = db
        self.updater = self._initialize_bot(bot_token)
        self.dispatcher = self.updater.dispatcher
        self._update_bot_data()

        conv_handler = ConversationHandler(
            entry_points=[CommandHandler('start', self.start),
                          CallbackQueryHandler(self.restart_state),
                          MessageHandler(Filters.all, self.not_recognized)],
            states={
                CHOOSE_PAIR_TF: [CallbackQueryHandler(self.select_base, pattern='^' + 'change_pair' + '$'),
                                 CallbackQueryHandler(self.select_tf, pattern='^' + 'change_tf' + '$')],
                CHOOSE_BASE_CURRENCY_STATE: [CallbackQueryHandler(self.cancel, pattern='^' + 'cancel' + '$'),
                                             CallbackQueryHandler(self.select_target)],
                CHOOSE_TARGET_CURRENCY_STATE: [CallbackQueryHandler(self.cancel, pattern='^' + 'cancel' + '$'),
                                               CallbackQueryHandler(self.done_pair)],
                CHOOSE_TIME_FRAME_STATE: [CallbackQueryHandler(self.cancel, pattern='^' + 'cancel' + '$'),
                                          CallbackQueryHandler(self.done_tf)],
            },
            fallbacks=[CommandHandler('start', self.start),
                       CallbackQueryHandler(self.restart_state),
                       MessageHandler(Filters.all, self.not_recognized)]
        )

        self.dispatcher.add_handler(conv_handler)
        self.dispatcher.add_error_handler(self.error)

    def run(self):
        logger.info('listening ...')
        self.updater.start_polling()

        # Run the bot until the user presses Ctrl-C or the process receives SIGINT,
        # SIGTERM or SIGABRT
        # self.updater.idle()

    def _update_bot_data(self):
        base_currency_list = BASE_CURRENCY_LIST
        target_currency_list = TARGET_CURRENCY_LIST
        time_frames = TIME_FRAMES

        self.dispatcher.bot_data.update({'base_currency_list': base_currency_list,
                                         'target_currency_list': target_currency_list,
                                         'time_frames': time_frames,
                                         'start_conversation': CHOOSE_PAIR_TF,
                                         'choose_base_currency': CHOOSE_BASE_CURRENCY_STATE,
                                         'choose_target_currency': CHOOSE_TARGET_CURRENCY_STATE,
                                         'choose_time_frame': CHOOSE_TIME_FRAME_STATE})

    @staticmethod
    def _initialize_bot(bot_token):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        persistent = PicklePersistence(filename=os.path.join(dir_path, 'saita_bot_data.pkl'))
        updater = Updater(str(bot_token),
                          persistence=persistent,
                          use_context=True,
                          workers=0)
        me = updater.bot.get_me()
        logger.info('{} started.'.format(me['username']))
        return updater

    @staticmethod
    def start(update, context):
        user = update.message.from_user
        chat_id = update.message.chat.id
        logger.info('received /start from: {}'.format(user['id']))

        if not db_handler.exists(user['id']):
            user_data = db_handler.add_user(user['id'],
                                            user['first_name'],
                                            user['last_name'],
                                            user['username'],
                                            chat_id)
        else:
            user_data = db_handler.get_user(user['id'])

        reply_text, reply_markup = get_start_markup(user_data.pair, user_data.time_frame)

        update.message.reply_text(reply_text,
                                  reply_markup=reply_markup,
                                  parse_mode=ParseMode.MARKDOWN)

        return CHOOSE_PAIR_TF

    @staticmethod
    def select_base(update, context):

        """The change_pair button has been selected and the conversation state is at CHOOSE_PAIR_TF"""

        query = update.callback_query
        user = query.from_user
        user_id = user['id']

        logger.info('{} selected change_pair'.format(user_id))

        query.answer('Changing the pair!')

        reply_text = """ارز مورد نظر رو انتخاب کن:"""

        keyboard = _get_base_currency_keyboard(context.bot_data['base_currency_list'])
        reply_markup = InlineKeyboardMarkup(keyboard)

        query.edit_message_text(
            text=reply_text,
            reply_markup=reply_markup,
            parse_mode='MarkdownV2'
        )
        return CHOOSE_BASE_CURRENCY_STATE

    @staticmethod
    def select_target(update, context):

        """The base currency has been selected and the conversation state is at CHOOSE_BASE_CURRENCY_STATE"""

        query = update.callback_query
        user = query.from_user
        user_id = user['id']
        selected_base = query.data

        logger.info('{} selected {} as base currency.'.format(user_id, selected_base))

        query.answer('{}!'.format(selected_base))

        reply_text = """ارز مبنا رو انتخاب کن:"""

        target_currency_list = list(context.bot_data['target_currency_list'])
        if selected_base.upper() in target_currency_list:
            target_currency_list.remove(selected_base.upper())
        # target_currency_list = set(target_currency_list)

        keyboard = _get_target_currency_keyboard(target_currency_list, selected_base)
        reply_markup = InlineKeyboardMarkup(keyboard)

        query.edit_message_text(
            text=reply_text,
            reply_markup=reply_markup,
            parse_mode='MarkdownV2'
        )
        return CHOOSE_TARGET_CURRENCY_STATE

    @staticmethod
    def done_pair(update, context):

        """The target currency has been selected and the conversation state is at CHOOSE_TARGET_CURRENCY_STATE.

        The state after this callback will be CHOOSE_PAIR_TF, i.e. the state after the /start command.
        """

        query = update.callback_query
        user = query.from_user
        user_id = user['id']

        selected_base, selected_target = query.data.split('_')
        selected_base = selected_base.upper()
        selected_target = selected_target.upper()

        logger.info('{} selected {}'.format(user_id, selected_target))

        query.answer('{}/{}! lets go!'.format(selected_base, selected_target))

        try:
            user_data = db_handler.update_currency_pair(selected_base, selected_target, user_id)
        except Exception as e:
            logger.exception(e)
            return

        reply_text, reply_markup = get_start_markup(user_data.pair, user_data.time_frame)

        query.edit_message_text(
            text=reply_text,
            reply_markup=reply_markup,
            parse_mode='MarkdownV2'
        )
        return CHOOSE_PAIR_TF

    @staticmethod
    def select_tf(update, context):

        """The change_tf button has been selected and the conversation state is at CHOOSE_PAIR_TF

        The state after this callback will be CHOOSE_PAIR_TF, i.e. the state after the /start command.
        """

        query = update.callback_query
        user = query.from_user
        user_id = user['id']

        logger.info('{} selected change_tf'.format(user_id))

        query.answer('Changing the timeframe!')

        reply_text = """قالب زمانی مورد نظر رو انتخاب کن:"""

        tfs = [i.string for i in context.bot_data['time_frames']]
        keyboard = _get_tf_keyboard(tfs)
        # keyboard = _get_tf_keyboard(context.bot_data['time_frames'])
        reply_markup = InlineKeyboardMarkup(keyboard)

        query.edit_message_text(
            text=reply_text,
            reply_markup=reply_markup,
            parse_mode='MarkdownV2'
        )
        return CHOOSE_TIME_FRAME_STATE

    @staticmethod
    def done_tf(update, context):

        """The time_frame has been selected and the conversation state is at CHOOSE_TIME_FRAME_STATE"""

        query = update.callback_query
        user = query.from_user
        user_id = user['id']
        selected_time_frame = query.data

        logger.info('{} selected timeframe {}'.format(user_id, selected_time_frame))

        query.answer('{}! lets go!'.format(selected_time_frame))

        try:
            user_data = db_handler.update_time_frame(selected_time_frame, user_id)
        except Exception as e:
            logger.exception(e)
            return

        reply_text, reply_markup = get_start_markup(user_data.pair, user_data.time_frame)

        query.edit_message_text(
            text=reply_text,
            reply_markup=reply_markup,
            parse_mode='MarkdownV2'
        )
        return CHOOSE_PAIR_TF

    @staticmethod
    def cancel(update, context):

        """Cancel button."""

        query = update.callback_query
        user = query.from_user
        user_id = user['id']

        logger.info('{} selected cancel.'.format(user_id))

        query.answer('OK!')

        user_data = db_handler.get_user(user_id)

        reply_text, reply_markup = get_start_markup(user_data.pair, user_data.time_frame)

        query.edit_message_text(
            text=reply_text,
            reply_markup=reply_markup,
            parse_mode='MarkdownV2'
        )
        return CHOOSE_PAIR_TF

    @staticmethod
    def restart_state(update, context):

        """Edits message's text and inline keyboards in order to have exactly the same format of result of /start,
         and returns CHOOSE_PAIR_TF as state.

         This is the case when a fallback of type callback_query has been occured.
         """

        query = update.callback_query
        user = query.from_user
        user_id = user['id']

        logger.info('restart state for {}'.format(user_id))

        if db_handler.exists(user_id):
            query.answer('lets do it!')
            user_data = db_handler.get_user(user_id)
            reply_text, reply_markup = get_start_markup(user_data.pair, user_data.time_frame)

            try:
                query.edit_message_text(
                    text=reply_text,
                    reply_markup=reply_markup,
                    parse_mode='MarkdownV2'
                )
            except BadRequest:
                pass  # Message has not modified
            return CHOOSE_PAIR_TF
        else:
            query.answer("I can't recognize you!")
            try:
                query.edit_message_text(
                    text="با ارسال /start شروع کنید.",
                    reply_markup=None,
                    parse_mode='MarkdownV2'
                )
            except BadRequest:
                pass  # Message has not modified
            return -1

    @staticmethod
    def not_recognized(update, context):

        """When the user is in the conversation and sends a message other than valid commands.

        It won't return anything in order to leave the state unchanged.
        """

        text = """متوجه نشدم! با دستور /start شروع کن."""
        update.message.reply_text(text)

    @staticmethod
    def error(update, context):

        """Log Errors caused by Updates."""

        logger.error('Update {} caused error {}'.format(update, context.error))


def _get_base_currency_keyboard(base_currency_list, ncols=4):
    count = len(base_currency_list) + 1
    keyboard = list()
    for s in range(0, count, ncols):
        e = s + ncols
        if e > count:
            e = count
        row = [InlineKeyboardButton(item, callback_data=item.lower()) for item in base_currency_list[s: e]]
        keyboard.append(row)
    keyboard.append([InlineKeyboardButton("Cancel", callback_data='cancel')])
    return keyboard


def _get_tf_keyboard(time_frames, ncols=3):
    count = len(time_frames) + 1
    keyboard = list()
    for s in range(0, count, ncols):
        e = s + ncols
        if e > count:
            e = count
        row = [InlineKeyboardButton(item, callback_data=item) for item in time_frames[s: e]]
        keyboard.append(row)
    keyboard.append([InlineKeyboardButton("Cancel", callback_data='cancel')])
    return keyboard


def _get_target_currency_keyboard(target_currency_list, selected_base, ncols=2):
    count = len(target_currency_list) + 1
    keyboard = list()
    for s in range(0, count, ncols):
        e = s + ncols
        if e > count:
            e = count
        row = [InlineKeyboardButton('{}'.format(item),
                                    callback_data='{}_{}'.format(selected_base,
                                                                 item.lower())) for item in target_currency_list[s: e]]
        keyboard.append(row)
    keyboard.append([InlineKeyboardButton("Cancel", callback_data='cancel')])
    return keyboard


def get_start_markup(pair, time_frame):
    text = '''تنظیمات کنونی:

*{}*@*{}*'''.format(pair,
                    time_frame)
    keyboard = [[InlineKeyboardButton("تغییر ارز مورد نظر", callback_data='change_pair')],
                [InlineKeyboardButton("تغییر قالب زمانی", callback_data='change_tf')]]

    markup = InlineKeyboardMarkup(keyboard)
    return text, markup


# if __name__ == '__main__':
#     token = '1069900023:AAGU8F0vdcAYxewlhbzsK8hxmfkggqqkbgs'
#     bot = SAITABot(token)
#     bot.run()
