require "v8"
require "json"

class V8::Object
  def to_json
    @context['JSON']['stringify'].call(self)
  end

  def to_hash
    JSON.parse(to_json, :max_nesting => false)
  end
end