require 'nokogiri'

module Jekyll
  module LunrJsSearch
    class PageRenderer
      def initialize(site)
        @site = site
      end
      
      # render item, but without using its layout
      def prepare(item)
        layout = item.data["layout"]
        begin
          item.data.delete("layout")

          if item.is_a?(Jekyll::Document)          
            output = Jekyll::Renderer.new(@site, item).run
          else
            item.render({}, @site.site_payload)
            output = item.output  
          end
        ensure
          # restore original layout
          item.data["layout"] = layout
        end
      
        output
      end

      # render the item, parse the output and get all text inside <p> elements
      def render(item)
        layoutless = item.dup

        Nokogiri::HTML(prepare(layoutless)).text
      end
    end
  end  
end