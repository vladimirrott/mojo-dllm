"""A small flag parser: `--name value` pairs plus boolean `--flag`s."""

from std.collections import Dict


struct Args(Movable):
    var command: String
    var values: Dict[String, String]
    var flags: List[String]
    var positional: List[String]

    def __init__(
        out self, argv: List[String], boolean_flags: List[String]
    ) raises:
        self.values = Dict[String, String]()
        self.flags = List[String]()
        self.positional = List[String]()
        self.command = argv[1] if len(argv) > 1 else String("")
        var i = 2
        while i < len(argv):
            var a = argv[i]
            if a.startswith("--"):
                var name = String(a[byte=2:])
                var is_bool = False
                for f in boolean_flags:
                    if f == name:
                        is_bool = True
                if is_bool:
                    self.flags.append(name)
                    i += 1
                    continue
                if i + 1 >= len(argv):
                    raise Error("flag --" + name + " needs a value")
                self.values[name] = argv[i + 1]
                i += 2
            else:
                self.positional.append(a)
                i += 1

    def has(self, name: String) -> Bool:
        if name in self.values:
            return True
        for f in self.flags:
            if f == name:
                return True
        return False

    def get(self, name: String) raises -> String:
        if name not in self.values:
            raise Error("missing required flag --" + name)
        return self.values[name]

    def get_or(self, name: String, default: String) raises -> String:
        if name in self.values:
            return self.values[name]
        return default

    def int_or(self, name: String, default: Int) raises -> Int:
        if name in self.values:
            return Int(self.values[name])
        return default

    def float_or(self, name: String, default: Float64) raises -> Float64:
        if name in self.values:
            return Float64(self.values[name])
        return default


def parse_csv_ints(s: String) raises -> List[Int]:
    var out = List[Int]()
    for part in s.split(","):
        var p = String(part).strip()
        if p.byte_length() > 0:
            out.append(Int(p))
    return out^
