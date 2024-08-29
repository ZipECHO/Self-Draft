class InferProfile:
    def __init__(self, profile_keys=None):
        profile_keys = [] if profile_keys is None else profile_keys
        self.pf = {}
        for key in profile_keys:
            self.pf[key] = 0

    def update_profile(self, key, value):
        # Update the value of the profile key
        if key in self.pf:
            self.pf[key] = value
        else:
            print(f"Key {key} does not exist in the profile.")

    def set_profile(self, key, value):
        self.pf[key] = value

    def get_profile(self, key):
        # Retrieve the value of the profile key
        if key in self.pf:
            return self.pf[key]
        else:
            # print(f"Key {key} does not exist in the profile.")
            return 0

    def incremental_update(self, key, value):
        # Increment the value of the profile key
        if key in self.pf:
            self.pf[key] += value
        else:
            self.pf[key] = value

    def incremental_updates(self, new_profile):
        # Increment the values of the profile keys in the new_profile dictionary
        if isinstance(new_profile, dict):
            for key, val in new_profile.items():
                self.incremental_update(key, val)
        elif isinstance(new_profile, InferProfile):
            for key, val in new_profile.pf.items():
                self.incremental_update(key, val)

        elif "InferProfile" in str(type(new_profile)):
            for key, val in new_profile.pf.items():
                self.incremental_update(key, val)
        elif new_profile is None:
            return

        else:
            raise RuntimeError('Unsupported type of new_profile')

    def __str__(self):
        # Return a well formated string representation of the pf variable
        s = '\n'
        for key, val in self.pf.items():
            s += f'{key}\t\t{val}\n'
        return s

    def output(self,keys):
        res = '\n'
        for key in keys:
            res += f'{key}\t\t{self.get_profile(key)}\n'
        return res

