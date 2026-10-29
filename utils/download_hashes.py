"""SHA-256 of the bootstrap archives Setup downloads.

Setup runs the portable Python and uv it downloads, so their archives are
checked against these values before extraction, as the model weights are.
Taken from the projects' published checksums (python-build-standalone
release 20241219 SHA256SUMS; uv 0.10.6 .sha256 files), cross-checked on
2026-10-29 against downloaded Windows archives. Update together with
python_manager.RELEASE_TAG / PYTHON_VERSIONS and uv_manager.UV_VERSION.
"""

import hashlib

# fmt: off
PYTHON_STANDALONE_SHA256 = {
    "cpython-3.10.16+20241219-aarch64-apple-darwin-install_only.tar.gz":
        "73af24f647b5cef9b9786bc140fca4b5615175a04a6f20a1726c3c19d27e9469",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.10.16+20241219-aarch64-unknown-linux-gnu-install_only.tar.gz":
        "1dd971c620e50f2d3ad72ddc58cc79cd01e5b83c6e653e4548f1aac1a52d9842",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.10.16+20241219-x86_64-apple-darwin-install_only.tar.gz":
        "a79edcd14cb4a48e1ec096eaa8160b35a6b03ba01f3c6f0dd724d5319771fe9a",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.10.16+20241219-x86_64-pc-windows-msvc-install_only.tar.gz":
        "10f7221c5eaa1bb78c24aeff00a4a34234ca8ec8ae038c4052acfdd65ab71966",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.10.16+20241219-x86_64-unknown-linux-gnu-install_only.tar.gz":
        "3e4538e9a57a23a16645748113dd173ccaaf17c456b3985a0995f1939ec22206",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.11.11+20241219-aarch64-apple-darwin-install_only.tar.gz":
        "8c12ffb8c5b98ee0bd7ab2d422b6a69a0a341bad0b8958f7ae119cd2ee954902",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.11.11+20241219-aarch64-unknown-linux-gnu-install_only.tar.gz":
        "4f8131bc1471f3fcde830d6cec789f009410cbf082be92942c029f62d1deff26",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.11.11+20241219-x86_64-apple-darwin-install_only.tar.gz":
        "9220f526950936a6306f19392584e3dac7033b60467a09f86cff6b9c1d970841",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.11.11+20241219-x86_64-pc-windows-msvc-install_only.tar.gz":
        "4b85ede0b442d7832a97f592843b51d8e7e4ecc7c4723914b391289bed321cf7",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.11.11+20241219-x86_64-unknown-linux-gnu-install_only.tar.gz":
        "579aa5409eb2d10df075baa9b4d53fef52a6e3f8b991907a54a25525f1f1151c",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.12.8+20241219-aarch64-apple-darwin-install_only.tar.gz":
        "f6a470078ddebdecfc94c8de65d4cab8ac06ef0d78e5388d3499463c0dccca93",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.12.8+20241219-aarch64-unknown-linux-gnu-install_only.tar.gz":
        "a62471474659e7aed1f96a5526d535f5b04fe0db631992974b5c10ee7c0e2999",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.12.8+20241219-x86_64-apple-darwin-install_only.tar.gz":
        "ab7a99c78c8afaf74e4ee2d2c696bb47b08ac76836608279729665a9162feb46",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.12.8+20241219-x86_64-pc-windows-msvc-install_only.tar.gz":
        "de7e417d1abc46f187790a06a1a691532793f509e0b7f8be79cec25c5fb5f108",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.12.8+20241219-x86_64-unknown-linux-gnu-install_only.tar.gz":
        "e172c063cdad4ab016f78834b1ef9fedfac80aae08918188fe6ebb3b325ce347",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.13.1+20241219-aarch64-apple-darwin-install_only.tar.gz":
        "a67dd58f6290f7b932620b955145f39eaad0d240328164233a63583edccb78cb",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.13.1+20241219-aarch64-unknown-linux-gnu-install_only.tar.gz":
        "c991fe7da0fdcdc35738169c02b966d23e69dbf3e33a6dd73d157fc9d14325ad",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.13.1+20241219-x86_64-apple-darwin-install_only.tar.gz":
        "ca0a5df026b84cb93a57bad021ff0fdc7c877cc0ce7e47b49a718689148a6067",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.13.1+20241219-x86_64-pc-windows-msvc-install_only.tar.gz":
        "d465fea3da068a0de9161abcd2ef2a9ae16c3d15fe3655a533b60fb6ac1732c6",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.13.1+20241219-x86_64-unknown-linux-gnu-install_only.tar.gz":
        "a5b258f032085bd4bd5c7c5265b52144991051ef77048a0405de80e2608ec2ec",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.9.21+20241219-aarch64-apple-darwin-install_only.tar.gz":
        "9d54cf4a7367ae0f87d273d61a44f89315b85981e7d53d60fbbe96e9b73d874d",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.9.21+20241219-aarch64-unknown-linux-gnu-install_only.tar.gz":
        "4cc3eb044592b3543da21934c3464492bef976998a6fc341822e60a42368f777",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.9.21+20241219-x86_64-apple-darwin-install_only.tar.gz":
        "6985f0c0428629600782b7a0c7917b1e07730f7b2e99f0b49b949dbed82a285d",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.9.21+20241219-x86_64-pc-windows-msvc-install_only.tar.gz":
        "58a2dee926d2ab9c7189d44133a19fd549ce34da0c0e69d833eb57c21e56b27d",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.9.21+20241219-x86_64-unknown-linux-gnu-install_only.tar.gz":
        "561a97bd53303ea6c1e6d4e8fc8b736e584d6950d502331ff9d03ca5446ce744",  # noqa: E501  # pragma: allowlist secret
}

UV_SHA256 = {
    "uv-aarch64-apple-darwin.tar.gz":
        "3993249d8f51deaf34cfce037e57e294e82267ff1f9dc45b7983a17afaf065b4",  # noqa: E501  # pragma: allowlist secret
    "uv-aarch64-unknown-linux-gnu.tar.gz":
        "9380705294a85e3e634570abddd5b2577900c1873c29b790c7abc56a81dce4bc",  # noqa: E501  # pragma: allowlist secret
    "uv-x86_64-apple-darwin.tar.gz":
        "d7647571fb17a5107d4d23cc190418039c157fd7361ddb59bc6f8127a49e3eac",  # noqa: E501  # pragma: allowlist secret
    "uv-x86_64-pc-windows-msvc.zip":
        "b27eb789f281e398a82197477de727fc8faf08605152115686da2c3cba0d25f7",  # noqa: E501  # pragma: allowlist secret
    "uv-x86_64-unknown-linux-gnu.tar.gz":
        "aaa402e19d14a6b9a4267fcf4ec35380f804c68923525cea67cd6ee05bb4e930",  # noqa: E501  # pragma: allowlist secret
}
# fmt: on


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_archive(path: str, filename: str, table: dict) -> str:
    """"" when ``path`` has the pinned SHA-256 of ``filename``, else the reason."""
    expected = table.get(filename)
    if expected is None:
        return f"no pinned checksum for {filename}"
    actual = sha256_of(path)
    if actual != expected:
        return (f"checksum mismatch for {filename} (expected {expected[:12]}..., "
                f"got {actual[:12]}...)")
    return ""
