"""Local administrative provisioning; no open registration HTTP endpoint."""

import argparse

from database.connection import init_db
from database.collaboration import (provision_user, issue_token, revoke_token,
                                    link_patient, set_user_active)


def main():
    parser = argparse.ArgumentParser(description="管理医生/家属账号、患者授权及站内API令牌")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create-user")
    create.add_argument("user_id")
    create.add_argument("role", choices=("doctor", "family"))
    token = sub.add_parser("issue-token")
    token.add_argument("user_id")
    token.add_argument("--hours", type=float, default=24)
    for name in ("grant", "revoke-grant"):
        command = sub.add_parser(name)
        command.add_argument("user_id")
        command.add_argument("patient_id")
    for name in ("disable-user", "enable-user"):
        command = sub.add_parser(name)
        command.add_argument("user_id")
    sub.add_parser("revoke-token")  # Read credential interactively, not from process arguments.
    args = parser.parse_args()
    init_db()
    if args.command == "create-user":
        provision_user(args.user_id, args.role)
    elif args.command == "issue-token":
        print(issue_token(args.user_id, expires_in_hours=args.hours))
        return
    elif args.command in {"grant", "revoke-grant"}:
        link_patient(args.user_id, args.patient_id, args.command == "grant")
    elif args.command in {"disable-user", "enable-user"}:
        set_user_active(args.user_id, args.command == "enable-user")
    else:
        from getpass import getpass
        revoke_token(getpass("Token to revoke: "))
    print("操作已保存。")


if __name__ == "__main__":
    main()
