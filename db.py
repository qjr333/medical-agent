"""Compatibility entry point for the original SQLite module.

Application repositories use :mod:`database.connection`. These wrappers preserve
legacy imports and explicit overrides of this module's DB_PATH.
"""

from database import connection as _connection

BASE_DIR = _connection.BASE_DIR
DATA_DIR = _connection.DATA_DIR
DB_PATH = _connection.DB_PATH


def get_connection():
    return _connection.get_connection(DB_PATH)


def init_db():
    return _connection.init_db(DB_PATH)


if __name__ == "__main__":
    init_db()
    print("SQLite数据库初始化成功！")
    print("数据库位置：", DB_PATH)
