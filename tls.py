"""
tls.py - self-signed HTTPS certificate for the dev server.

Phones only let a web page use the camera on HTTPS (or on localhost, which a phone is not), so to try
battle mode from your phone on your own Wi-Fi run:  python app.py --host 0.0.0.0 --https
and open the https://<your-computer's-IP>:5000/detect (or /battle) address it prints. The phone's browser warns about
the certificate (it is signed by nobody but you) - choose "advanced / continue". The certificate is valid
for localhost, 127.0.0.1 and this machine's LAN addresses, and is re-made when those change.

BACKGROUND
  HTTPS needs a *certificate*: a file that says "this public key belongs to this address". Normally a trusted
  authority signs it. Here we sign it ourselves ("self-signed"), so browsers show a warning - but the traffic is
  still encrypted and, crucially, the browser now treats the page as a "secure context", which unlocks the
  camera API (navigator.mediaDevices). The addresses the certificate is valid for are listed in its
  "Subject Alternative Name" (SAN) extension - that is why the IPs are baked in and the cert is rebuilt when they change.
"""
import datetime, ipaddress, os, socket
from pathlib import Path

CERT_DIR = Path(__file__).parent / "certs"


def lan_ips():
    """This machine's non-loopback IPv4 addresses (no network traffic is sent).

    Trick: a UDP socket "connected" to an address does not send anything, but the operating system must pick
    which network interface would be used - and getsockname() then tells us that interface's IP."""
    ips = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))                 # UDP "connect" only picks the outgoing interface
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    try:
        ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def _wanted_names(extra=()):
    return {"localhost", "127.0.0.1", *lan_ips(), *extra}


def _cert_names(cert):
    from cryptography import x509
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    return {str(v) for v in san.get_values_for_type(x509.IPAddress)} | set(san.get_values_for_type(x509.DNSName))


def ensure_cert(directory=CERT_DIR, extra_hosts=()):
    """Return (cert_path, key_path), creating or renewing the certificate when needed.

    Steps: (1) reuse the existing cert if it covers every address we need and has >14 days left;
    (2) otherwise generate a 2048-bit RSA private key; (3) build a certificate whose subject and issuer are
    the same (self-signed), valid for 1 year, with the SAN list; (4) save both as PEM text files
    (the key file is made readable by the owner only)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cert_path, key_path = directory / "pokecard-dev-cert.pem", directory / "pokecard-dev-key.pem"
    names = _wanted_names(extra_hosts)

    if cert_path.exists() and key_path.exists():
        try:
            cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
            left = cert.not_valid_after_utc - datetime.datetime.now(datetime.timezone.utc)
            if names <= _cert_names(cert) and left > datetime.timedelta(days=14):
                return str(cert_path), str(key_path)
        except Exception:
            pass                                             # unreadable / old format: make a new one

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    alt = []
    for n in sorted(names):
        try:
            alt.append(x509.IPAddress(ipaddress.ip_address(n)))
        except ValueError:
            alt.append(x509.DNSName(n))
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "PokeCard 3D dev server")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(alt), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .sign(key, hashes.SHA256()))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                           serialization.NoEncryption()))
    try:
        os.chmod(key_path, 0o600)
    except OSError:
        pass
    return str(cert_path), str(key_path)
