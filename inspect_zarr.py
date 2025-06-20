from zarr.storage import FSStore
import zarr

def inspect_zarr(zarr_path):
    from zarr.storage import FSStore
    store = FSStore(str(zarr_path), dimension_separator="/")
    z = zarr.group(store=store)
    
    print("✅ Zarr v3 store opened successfully!")
    print("📂 Zarr Group Tree:\n")
    print(z.tree())

    def recurse(group, prefix=""):
        found_any = False
        for key, val in group.items():
            found_any = True
            if isinstance(val, zarr.hierarchy.Group):
                print(f"\n🔸 Group: {prefix + key}")
                print("Attributes:", val.attrs.asdict())
                recurse(val, prefix + key + "/")
            else:
                print(f"📈 Array: {prefix + key} — shape {val.shape}, dtype {val.dtype}")
                print("Attributes:", val.attrs.asdict())
        if not found_any:
            print(f"⚠️ No subgroups or arrays found in: {prefix or '/'}")

    recurse(z)
