"""FastAPI dependencies using server-issued opaque bearer credentials only."""

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from database.collaboration import authenticate_token, current_user
from database.collaboration import require_patient_access as _require_patient_access

_bearer = HTTPBearer(auto_error=False)


def authenticated_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer)):
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="需要有效的登录令牌", headers={"WWW-Authenticate": "Bearer"})
    user = authenticate_token(credentials.credentials)
    if user is None:
        raise HTTPException(status_code=401, detail="登录令牌无效或已过期", headers={"WWW-Authenticate": "Bearer"})
    return user


def require_role(user, *roles):
    try:
        identity = current_user(user)
        if identity["role"] not in roles:
            raise PermissionError("Role is not permitted")
        return identity
    except PermissionError:
        raise HTTPException(status_code=403, detail="无权访问此功能") from None


def require_patient_access(user, patient_id, *roles):
    try:
        return _require_patient_access(user, patient_id, roles=roles or ("doctor", "family"))
    except PermissionError:
        raise HTTPException(status_code=403, detail="未获授权访问该患者") from None
