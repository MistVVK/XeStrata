"""WSGI-compatible langid web service (`langid -s` serves it)."""

import json
from http import HTTPStatus
from urllib.parse import parse_qs

from .langid import classify, rank


def _detect(data):
    lang, conf = classify(data)
    return {'language': lang, 'confidence': conf}


_ROUTES = {'detect': _detect, 'rank': rank}


def application(environ, start_response):
    """WSGI-compatible langid web service."""
    path = environ.get('PATH_INFO', '').strip('/').partition('/')[0]
    handler = _ROUTES.get(path)
    if handler is None:
        return _return_response(start_response, 404, None, 'Not found')

    method = environ['REQUEST_METHOD']
    if method not in ('GET', 'POST', 'PUT'):
        return _return_response(start_response, 405, None, f'{method} not allowed')

    data = _get_data(environ, method)
    if data is None:
        return _return_response(start_response, 400, None, 'No data provided')

    return _return_response(start_response, 200, handler(data), None)


def _get_data(environ, method):
    if method == 'GET':
        try:
            return parse_qs(environ.get('QUERY_STRING', ''))['q'][0]
        except KeyError:
            return None
    try:
        length = int(environ.get('CONTENT_LENGTH', 0))
    except ValueError:
        return None
    if length <= 0:
        return None
    data = environ['wsgi.input'].read(length)
    if method == 'POST':
        try:
            data = parse_qs(data)[b'q'][0]
        except KeyError:
            pass
    return data


def _return_response(start_response, status_code, response_data, response_details):
    status = HTTPStatus(status_code)
    response = {
        'responseData': response_data,
        'responseStatus': status_code,
        'responseDetails': response_details,
    }
    headers = [('Content-type', 'application/json; charset=utf-8')]
    start_response(f"{status.value} {status.phrase}", headers)
    return [json.dumps(response).encode('utf-8')]
