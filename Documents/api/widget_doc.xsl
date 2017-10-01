<?xml version='1.0'?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
<xsl:template match="/">
  <html>
  <body>
    <h1>Veusz widget API</h1>
      <xsl:for-each select="widgets/widget">
        <h2>
	  <a href="#{apiname}"></a>
	  <xsl:value-of select="apiname"/>
	</h2>
	<div><xsl:value-of select="description"/></div>
	<div>Can be placed in:
	  <xsl:for-each select="allowedparent">
	    <xsl:apply-templates/><xsl:text> </xsl:text>
	  </xsl:for-each>
	</div>
	<!-- iterate over settings in widget -->
	<xsl:for-each select="settings">
	  <h3><xsl:value-of select="apiname"/></h3>
	    <p>Display name:
	      <xsl:value-of select="displayname"/>,
	      description: 
	      <xsl:value-of select="description"/>
	    </p>

	  <table border="1">
	    <tr>
	      <th>Setting API name</th><th>Display name</th>
	      <th>Description</th><th>Type</th><th>Default</th>
	      <th>Choice</th>
	    </tr>
	    <!-- iterate over setting in settings -->
	    <xsl:for-each select="setting">
	      <tr>
		<td><xsl:value-of select="apiname"/></td>
		<td><xsl:value-of select="displayname"/></td>
		<td><xsl:value-of select="description"/></td>
		<td><xsl:value-of select="type"/></td>
		<td><xsl:value-of select="default"/></td>
		<td>
		  <xsl:for-each select="choice">
		    <xsl:apply-templates/><xsl:text> </xsl:text>
		  </xsl:for-each>
		</td>
	      </tr>
	    </xsl:for-each>
	  </table>
	</xsl:for-each>
      </xsl:for-each>
  </body>
  </html>
</xsl:template>
</xsl:stylesheet>
