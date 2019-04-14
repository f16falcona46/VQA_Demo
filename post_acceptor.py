import wsgiref.simple_server
from urllib.parse import parse_qs, urlencode
import cgi, os, urllib.request, urllib

text = '''<! DOCTYPE html>
<html>
<body>
<form action="upload.php" method="post" enctype="multipart/form-data">
<input type="file" name="file" id="file">
<input type="text" name="q" id="q">
<input type="submit" value="Upload Image" name="submit">
</form>
</body>'''
def main():
    def app(env, start_response):
        start_response("200 OK", [("Content-type", "text/html; charset=utf-8")])
        q = parse_qs(env["QUERY_STRING"])
        fields = None
        if 'POST' == env['REQUEST_METHOD'] :        
            fields = cgi.FieldStorage(fp=env['wsgi.input'], environ=env, keep_blank_values=True)
            fileitem = fields['file']
            fn = os.path.basename(fileitem.filename) 
            open('uploads/' + fn, 'wb').write(fileitem.file.read())
            query = urlencode({'f': ("uploads/%s" % fileitem.filename), "q": fields.getlist('q')[0]})
            with urllib.request.urlopen("http://localhost:8080/?" + query) as r:
               contents = r.read()
            return [contents]
        
        return [text.encode('utf-8')]

    httpd = wsgiref.simple_server.make_server("localhost", 8081, app)
    httpd.serve_forever()
    
if __name__ == "__main__":
    main()
