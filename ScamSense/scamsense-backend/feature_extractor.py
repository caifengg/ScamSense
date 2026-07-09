import re
import ipaddress
from urllib.parse import urlparse

FEATURE_NAMES = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "HasObfuscation",
    "NoOfDegitsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "IsHTTPS",
]

# Characters that are structurally required in almost every URL - not
# counting these avoids flagging every "https://example.com" as suspicious
# just for having a scheme and a domain separator.
_NORMAL_URL_CHARS = set("/:._-?&=")


def extract_features(url: str) -> list:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    domain = parsed.netloc.split(":")[0]  # strip port if present

    try:
        ipaddress.ip_address(domain)
        is_domain_ip = 1
    except ValueError:
        is_domain_ip = 0

    has_obfuscation = 1 if ("%" in url or re.search(r"0x[0-9a-fA-F]+", url)) else 0

    special_chars = sum(
        (not c.isalnum()) and (c not in _NORMAL_URL_CHARS)
        for c in url
    )

    return [
        len(url),                                    # URLLength
        len(domain),                                  # DomainLength
        is_domain_ip,                                  # IsDomainIP
        has_obfuscation,                                # HasObfuscation
        sum(c.isdigit() for c in url),                  # NoOfDegitsInURL
        url.count("?"),                                 # NoOfQMarkInURL
        url.count("&"),                                 # NoOfAmpersandInURL
        special_chars,                                   # NoOfOtherSpecialCharsInURL
        1 if url.startswith("https") else 0,             # IsHTTPS
    ]